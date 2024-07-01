use ark_ff::PrimeField;

use ark_poly_commit::linear_codes::{
    calculate_t, get_indices_from_sponge, LinCodeParametersInfo, LinearCodePCS, LinearEncode,
};

use ark_crypto_primitives::crh::{CRHScheme, TwoToOneCRHScheme};
use ark_crypto_primitives::{
    merkle_tree::Config,
    sponge::{Absorb, CryptographicSponge},
};
use ark_poly::Polynomial;
use ark_poly_commit::{to_bytes, LabeledCommitment};
use ark_std::{borrow::Borrow, rand::RngCore};

use crate::{utils::inner_product, NaysayerError, PCSNaysayer};

#[cfg(not(feature = "std"))]
use ark_std::vec::Vec;

#[cfg(test)]
mod tests;

/// Naysayer proof for a single LinCodePCProof (corresponding to one opening),
/// indicating which piece of the opening proof is incorrect
#[derive(PartialEq, Debug, Clone)]
pub enum LinearCodeNaysayerProofSingle {
    /// Mismatch between the index of the Merkle path provided and the challenge
    /// index squeezed from the sponge
    PathIndexAssertion(usize),
    /// Incorrect Merkle path proof
    MerklePathAssertion(usize),
    /// Mismatch between E(vM) and vE(M) at a challenge position
    ColumnInnerProductAssertion(usize),
    /// Mismatch between claimed evaluation and t1 M t2, where t1 and t2 are the
    /// two halves of the tensor of the point (powers in the UV case, ML
    /// Lagrange basis evaluation in the ML case)
    EvaluationAssertion,
}

/// Naysayer proof for an LPCPArray proof (which is a vector of LinCodePCProof,
/// referring to multiple LinearCodePCS openings), which includes both the index
/// of the opening proof that is incorrect and the naysayer proof for that
/// opening
#[derive(PartialEq, Debug, Clone)]
pub struct LinearCodeNaysayerProof {
    pub incorrect_proof_index: usize,
    pub naysayer_proof_single: LinearCodeNaysayerProofSingle,
}

impl<L, F, P, C, H> PCSNaysayer<F, P> for LinearCodePCS<L, F, P, C, H>
where
    L: LinearEncode<F, C, P, H>,
    F: PrimeField + Absorb,
    P: Polynomial<F>,
    C: Config + 'static,
    Vec<F>: Borrow<<H as CRHScheme>::Input>,
    H::Output: Into<C::Leaf> + Send,
    C::Leaf: Sized + Clone + Default + Send + AsRef<C::Leaf>,
    H: CRHScheme + 'static,
{
    type NaysayerProof = LinearCodeNaysayerProof;

    /// Indicate whether the given proofs are valid or point to their issues,
    /// producing a naysayer proof
    fn naysay<'a>(
        vk: &Self::VerifierKey,
        coms: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = F>,
        proofs: &Self::Proof,
        sponge: &mut impl CryptographicSponge,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<Option<Self::NaysayerProof>, NaysayerError> {
        assert!(
            !vk.check_well_formedness(),
            "naysay is only implemented without the well-formedness check",
        );

        let coms = coms.into_iter().collect::<Vec<_>>();
        let values = values.into_iter().collect::<Vec<_>>();

        assert!(coms.len() == values.len());
        assert!(coms.len() == proofs.len());

        let leaf_hash_param: &<<C as Config>::LeafHash as CRHScheme>::Parameters =
            vk.leaf_hash_param();
        let two_to_one_hash_param: &<<C as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters =
            vk.two_to_one_hash_param();

        for (i, (labeled_commitment, value)) in coms.iter().zip(values).enumerate() {
            let proof = &proofs[i];
            let commitment = labeled_commitment.commitment();
            let n_rows = commitment.metadata.n_rows;
            let n_cols = commitment.metadata.n_cols;
            let n_ext_cols = commitment.metadata.n_ext_cols;
            let root = &commitment.root;
            let t = calculate_t::<F>(vk.sec_param(), vk.distance(), n_ext_cols)?;

            sponge
                .absorb(&to_bytes!(&commitment.root).map_err(|_| NaysayerError::TranscriptError)?);

            // 1. Seed the transcript with the point and the recieved vector
            // TODO Consider removing the evaluation point from the transcript.
            let point_vec = L::point_to_vec(point.clone());
            sponge.absorb(&point_vec);
            sponge.absorb(&proof.opening.v);

            // 2. Ask random oracle for the `t` indices where the checks happen.
            let indices = get_indices_from_sponge(n_ext_cols, t, sponge)?;

            // 3. Hash the received columns into leaf hashes.
            let col_hashes: Vec<C::Leaf> = proof
                .opening
                .columns
                .iter()
                .map(|c| {
                    H::evaluate(vk.col_hash_params(), c.clone())
                        .map_err(|_| NaysayerError::HashingError)
                        .unwrap()
                        .into()
                })
                .collect();

            // 4. Verify the paths for each of the leaf hashes - this is only run once,
            // even if we have a well-formedness check (i.e., we save sending and checking the columns).
            // See "Concrete optimizations to the commitment scheme", p.12 of [Brakedown](https://eprint.iacr.org/2021/1043.pdf).
            for (j, (leaf, q_j)) in col_hashes.iter().zip(indices.iter()).enumerate() {
                let path = &proof.opening.paths[j];
                if path.leaf_index != *q_j {
                    return Ok(Some(LinearCodeNaysayerProof {
                        incorrect_proof_index: i,
                        naysayer_proof_single: LinearCodeNaysayerProofSingle::PathIndexAssertion(j),
                    }));
                }

                if !path
                    .verify(leaf_hash_param, two_to_one_hash_param, root, leaf.clone())
                    .map_err(|_| NaysayerError::HashingError)?
                {
                    return Ok(Some(LinearCodeNaysayerProof {
                        incorrect_proof_index: i,
                        naysayer_proof_single: LinearCodeNaysayerProofSingle::MerklePathAssertion(
                            j,
                        ),
                    }));
                }
            }

            // 5. Compute the encoding w = E(v).
            let w = L::encode(&proof.opening.v, vk)?;

            // 6. Compute `a`, `b` to right- and left- multiply with the matrix `M`.
            let (a, b) = L::tensor(point, n_cols, n_rows);

            // 7. Probabilistic checks that whatever the prover sent,
            // matches with what the verifier computed for himself.
            // Note: we sacrifice some code repetition in order not to repeat execution.
            for (transcript_index, matrix_index) in indices.iter().enumerate() {
                if inner_product(&b, &proof.opening.columns[transcript_index]) != w[*matrix_index] {
                    return Ok(Some(LinearCodeNaysayerProof {
                        incorrect_proof_index: i,
                        naysayer_proof_single:
                            LinearCodeNaysayerProofSingle::ColumnInnerProductAssertion(
                                transcript_index,
                            ),
                    }));
                }
            }

            if inner_product(&proof.opening.v, &a) != value {
                return Ok(Some(LinearCodeNaysayerProof {
                    incorrect_proof_index: i,
                    naysayer_proof_single: LinearCodeNaysayerProofSingle::EvaluationAssertion,
                }));
            }
        }

        Ok(None)
    }

    fn verify_naysay<'a>(
        vk: &Self::VerifierKey,
        coms: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = F>,
        proofs: &Self::Proof,
        naysayer_proof: &Self::NaysayerProof,
        sponge: &mut impl CryptographicSponge,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, NaysayerError> {
        let leaf_hash_param: &<<C as Config>::LeafHash as CRHScheme>::Parameters =
            vk.leaf_hash_param();
        let two_to_one_hash_param: &<<C as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters =
            vk.two_to_one_hash_param();

        let LinearCodeNaysayerProof {
            incorrect_proof_index,
            naysayer_proof_single,
        } = naysayer_proof;
        let proof = &proofs[*incorrect_proof_index];
        let com = coms
            .into_iter()
            .nth(*incorrect_proof_index)
            .unwrap()
            .commitment();
        let value = values.into_iter().nth(*incorrect_proof_index).unwrap();

        let n_rows = com.metadata.n_rows;
        let n_cols = com.metadata.n_cols;
        let n_ext_cols = com.metadata.n_ext_cols;
        let root = &com.root;

        let t = calculate_t::<F>(vk.sec_param(), vk.distance(), n_ext_cols)?;

        match naysayer_proof_single {
            LinearCodeNaysayerProofSingle::PathIndexAssertion(j)
            | LinearCodeNaysayerProofSingle::MerklePathAssertion(j)
            | LinearCodeNaysayerProofSingle::ColumnInnerProductAssertion(j) => {
                if *j >= t {
                    return Ok(false);
                }

                match naysayer_proof_single {
                    LinearCodeNaysayerProofSingle::PathIndexAssertion(j) => {
                        sponge.absorb(
                            &to_bytes!(&com.root).map_err(|_| NaysayerError::TranscriptError)?,
                        );

                        // 1. Seed the transcript with the point and the recieved vector
                        // TODO Consider removing the evaluation point from the transcript.
                        let point_vec = L::point_to_vec(point.clone());
                        sponge.absorb(&point_vec);
                        sponge.absorb(&proof.opening.v);

                        // 2. Ask random oracle for the `t` indices where the checks happen.
                        let indices = get_indices_from_sponge(n_ext_cols, j + 1, sponge)?;

                        // check that the index is indeed wrong
                        let path = &proof.opening.paths[*j];
                        let q_j = indices[*j];
                        Ok(path.leaf_index != q_j)
                    }
                    LinearCodeNaysayerProofSingle::MerklePathAssertion(j) => {
                        let leaf =
                            H::evaluate(vk.col_hash_params(), proof.opening.columns[*j].clone())
                                .map_err(|_| NaysayerError::HashingError)?;
                        let path = &proof.opening.paths[*j];
                        let is_path_ok = path
                            .verify(leaf_hash_param, two_to_one_hash_param, root, leaf.into())
                            .map_err(|_| NaysayerError::HashingError)?;
                        Ok(!is_path_ok)
                    }
                    LinearCodeNaysayerProofSingle::ColumnInnerProductAssertion(j) => {
                        sponge.absorb(
                            &to_bytes!(&com.root).map_err(|_| NaysayerError::TranscriptError)?,
                        );

                        // 1. Seed the transcript with the point and the recieved vector
                        // TODO Consider removing the evaluation point from the transcript.
                        let point_vec = L::point_to_vec(point.clone());
                        sponge.absorb(&point_vec);
                        sponge.absorb(&proof.opening.v);

                        // 2. Ask random oracle for the `t` indices where the checks happen.
                        let indices = get_indices_from_sponge(n_ext_cols, j + 1, sponge)?;

                        // 5. Compute the encoding w = E(v).
                        let w = L::encode(&proof.opening.v, vk)?;

                        let (_, b) = L::tensor(point, n_cols, n_rows);

                        Ok(inner_product(&b, &proof.opening.columns[*j]) != w[indices[*j]])
                    }
                    _ => unreachable!(),
                }
            }
            LinearCodeNaysayerProofSingle::EvaluationAssertion => {
                let (a, _) = L::tensor(point, n_cols, n_rows);

                Ok(inner_product(&proof.opening.v, &a) != value)
            }
        }
    }
}
