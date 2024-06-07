use ark_ff::PrimeField;

use crate::linear_codes::{
    calculate_t, get_indices_from_sponge, LPCPArray, LinCodePCCommitment, LinCodeParametersInfo,
    LinearCodePCS, LinearEncode,
};

use crate::utils::inner_product;
use crate::{to_bytes, Error, LabeledCommitment};

use ark_crypto_primitives::crh::{CRHScheme, TwoToOneCRHScheme};
use ark_crypto_primitives::{
    merkle_tree::Config,
    sponge::{Absorb, CryptographicSponge},
};
use ark_poly::Polynomial;
use ark_std::borrow::Borrow;
use ark_std::rand::RngCore;

#[cfg(test)]
mod tests;

/// Naysayer proof for LinearCodePCS opening, indicating which piece of the
/// opening proof is incorrect (if any)
#[derive(PartialEq, Debug)]
pub enum LinearCodeNaysayerProof {
    /// No errors, opening proof accepted
    Aye,
    /// Mismatch between the index of the Merkle path provided and the challenge
    /// index squeezed from the sponge
    PathIndexLie(usize),
    /// Incorrect Merkle path proof
    MerklePathLie(usize),
    /// Mismatch between E(vM) and vE(M) at a challenge position
    ColumnInnerProductLie(usize),
    /// Mismatch between claimed evaluation and t1 M t2, where t1 and t2 are the
    /// two halves of the tensor of the point (powers in the UV case, ML
    /// Lagrange basis evaluation in the ML case)
    EvaluationLie,
}

impl<L, F, P, C, H> LinearCodePCS<L, F, P, C, H>
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
    /// Indicate whether the given proofs are valid or point to their issues,
    /// producing a naysayer proof
    pub fn naysay<'a>(
        vk: &L::LinCodePCParams,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<LinCodePCCommitment<C>>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = F>,
        proof_array: &LPCPArray<F, C>,
        sponge: &mut impl CryptographicSponge,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<LinearCodeNaysayerProof, Error> {
        assert!(
            !vk.check_well_formedness(),
            "naysay is only implemented without the well-formedness check",
        );

        let commitments: Vec<&LabeledCommitment<LinCodePCCommitment<C>>> =
            commitments.into_iter().collect();
        let values: Vec<F> = values.into_iter().collect();
        assert!(commitments.len() == values.len());
        assert!(commitments.len() == proof_array.len());
        assert!(commitments.len() == 1);

        assert!(
            commitments.len() == 1,
            "naysay is only implemented for opening proofs of a single polynomial",
        );

        let leaf_hash_param: &<<C as Config>::LeafHash as CRHScheme>::Parameters =
            vk.leaf_hash_param();
        let two_to_one_hash_param: &<<C as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters =
            vk.two_to_one_hash_param();

        let labeled_commitment = commitments[0];
        let value = values[0];

        let proof = &proof_array[0];
        let commitment = labeled_commitment.commitment();
        let n_rows = commitment.metadata.n_rows;
        let n_cols = commitment.metadata.n_cols;
        let n_ext_cols = commitment.metadata.n_ext_cols;
        let root = &commitment.root;
        let t = calculate_t::<F>(vk.sec_param(), vk.distance(), n_ext_cols)?;

        sponge.absorb(&to_bytes!(&commitment.root).map_err(|_| Error::TranscriptError)?);

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
                    .map_err(|_| Error::HashingError)
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
                return Ok(LinearCodeNaysayerProof::PathIndexLie(j));
            }

            if !path
                .verify(leaf_hash_param, two_to_one_hash_param, root, leaf.clone())
                .map_err(|_| Error::HashingError)?
            {
                return Ok(LinearCodeNaysayerProof::MerklePathLie(j));
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
                return Ok(LinearCodeNaysayerProof::ColumnInnerProductLie(
                    transcript_index,
                ));
            }
        }

        if inner_product(&proof.opening.v, &a) != value {
            return Ok(LinearCodeNaysayerProof::EvaluationLie);
        }

        Ok(LinearCodeNaysayerProof::Aye)
    }

    /// Verifies the naysayer proof.
    /// - Returns `Ok(true)` if the original proof is rejected (i.e. the
    ///   naysayer proof points to a valid issue).
    /// - Returns `Ok(false)` if the original proof is accepted, i.e.
    ///     - either the naysayer proof told to accept the original proof
    ///       ("Aye")
    ///     - or the naysayer proof points to an invalid issue
    /// - Returns `Err` if another type of error occurs during verification of
    ///   the naysayer proof.
    pub fn naysayer_verify<'a>(
        vk: &L::LinCodePCParams,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<LinCodePCCommitment<C>>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = F>,
        proof_array: &LPCPArray<F, C>,
        naysayer_proof: LinearCodeNaysayerProof,
        sponge: &mut impl CryptographicSponge,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, Error> {
        let leaf_hash_param: &<<C as Config>::LeafHash as CRHScheme>::Parameters =
            vk.leaf_hash_param();
        let two_to_one_hash_param: &<<C as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters =
            vk.two_to_one_hash_param();

        let commitments: Vec<&LabeledCommitment<LinCodePCCommitment<C>>> =
            commitments.into_iter().collect();
        let values: Vec<F> = values.into_iter().collect();
        assert!(commitments.len() == values.len());
        assert!(commitments.len() == proof_array.len());
        assert!(commitments.len() == 1);

        let proof = &proof_array[0];
        let commitment = commitments[0].commitment();
        let value = values[0];

        let n_rows = commitment.metadata.n_rows;
        let n_cols = commitment.metadata.n_cols;
        let n_ext_cols = commitment.metadata.n_ext_cols;
        let root = &commitment.root;

        let t = calculate_t::<F>(vk.sec_param(), vk.distance(), n_ext_cols)?;

        match naysayer_proof {
            LinearCodeNaysayerProof::Aye => Ok(false),
            LinearCodeNaysayerProof::PathIndexLie(j)
            | LinearCodeNaysayerProof::MerklePathLie(j)
            | LinearCodeNaysayerProof::ColumnInnerProductLie(j) => {
                if j >= t {
                    return Ok(false);
                }

                match naysayer_proof {
                    LinearCodeNaysayerProof::PathIndexLie(j) => {
                        sponge.absorb(
                            &to_bytes!(&commitment.root).map_err(|_| Error::TranscriptError)?,
                        );

                        // 1. Seed the transcript with the point and the recieved vector
                        // TODO Consider removing the evaluation point from the transcript.
                        let point_vec = L::point_to_vec(point.clone());
                        sponge.absorb(&point_vec);
                        sponge.absorb(&proof.opening.v);

                        // 2. Ask random oracle for the `t` indices where the checks happen.
                        let indices = get_indices_from_sponge(n_ext_cols, j + 1, sponge)?;

                        // check that the index is indeed wrong
                        let path = &proof.opening.paths[j];
                        let q_j = indices[j];
                        Ok(path.leaf_index != q_j)
                    }
                    LinearCodeNaysayerProof::MerklePathLie(j) => {
                        let leaf =
                            H::evaluate(vk.col_hash_params(), proof.opening.columns[j].clone())
                                .map_err(|_| Error::HashingError)?;
                        let path = &proof.opening.paths[j];
                        let is_path_ok = path
                            .verify(leaf_hash_param, two_to_one_hash_param, root, leaf.into())
                            .map_err(|_| Error::HashingError)?;
                        Ok(!is_path_ok)
                    }
                    LinearCodeNaysayerProof::ColumnInnerProductLie(j) => {
                        sponge.absorb(
                            &to_bytes!(&commitment.root).map_err(|_| Error::TranscriptError)?,
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

                        Ok(inner_product(&b, &proof.opening.columns[j]) != w[indices[j]])
                    }
                    _ => unreachable!(),
                }
            }
            LinearCodeNaysayerProof::EvaluationLie => {
                let (a, _) = L::tensor(point, n_cols, n_rows);

                Ok(inner_product(&proof.opening.v, &a) != value)
            }
        }
    }
}
