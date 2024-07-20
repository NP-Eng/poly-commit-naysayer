use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

use ark_crypto_primitives::{
    crh::CRHScheme,
    merkle_tree::Config,
    sponge::{Absorb, CryptographicSponge},
};
use ark_poly::Polynomial;

use ark_ff::{PrimeField};


use ark_poly_commit::{
    linear_codes::{
        calculate_t, create_merkle_tree, get_indices_from_sponge, LPCPArray, LinCodePCCommitment,
        LinCodePCCommitmentState, LinCodePCProof, LinCodePCProofSingle, LinCodeParametersInfo,
        LinearEncode,
    },
    to_bytes, LabeledCommitment,
};

use crate::{
    NaysayerError,
};

// Types of dishonesty that can be introduced in a LinearCodePCS proof
#[derive(Clone, PartialEq)]
pub enum LinearCodeDishonesty {
    // No dishonesty: same as the open
    None,
    // Modify v = bM after honestly producing a proof, leading to an
    // inconsistent sponge
    RowLCOutside,
    // Modify v = bM before feeding it to the sponge, leading to a consistent
    // sponge but inconsistent check E(b M) = b E(M)
    RowLCInside,
    // Modify one of the sent columns, leading to a failure in the Merkle path
    // verification (note that the verifier hashes the column into a leaf which
    // is used during path verification)
    Column,
    // Modify a node in one of the Merkle path proofs, leading to its incorrect
    // verification
    MerklePath,
    // Modify the leaf index in one of the Merkle path proofs, leading to a
    // mismatch with the query index squeezed from the sponge
    MerkleLeafIndex(usize),
    // Claim an evaluation y different from the actual one f(x)
    Evaluation,
}

// Generates a LinearCodePCS proof introducing a dishonesty
pub fn dishonest_open_lcpcs<L, F, P, C, H>(
    ck: &L::LinCodePCParams,
    com: &LabeledCommitment<LinCodePCCommitment<C>>,
    com_state: &LinCodePCCommitmentState<F, H>,
    point: &P::Point,
    sponge: &mut impl CryptographicSponge,
    dishonesty: LinearCodeDishonesty,
) -> Result<LPCPArray<F, C>, NaysayerError>
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
    let commitment = com.commitment();
    let n_rows = commitment.metadata.n_rows;
    let n_cols = commitment.metadata.n_cols;

    // 1. Arrange the coefficients of the polynomial into a matrix,
    // and apply encoding to get `ext_mat`.
    // 2. Create the Merkle tree from the hashes of each column.
    let LinCodePCCommitmentState {
        mat,
        ext_mat,
        leaves: col_hashes,
    } = com_state;
    let mut col_hashes: Vec<C::Leaf> = col_hashes.clone().into_iter().map(|h| h.into()).collect(); // TODO cfg_inter

    let col_tree = create_merkle_tree::<C>(
        &mut col_hashes,
        ck.leaf_hash_param(),
        ck.two_to_one_hash_param(),
    )?;

    // 3. Generate vector `b` to left-multiply the matrix.
    let (_, b) = L::tensor(point, n_cols, n_rows);

    sponge.absorb(&to_bytes!(&commitment.root).map_err(|_| NaysayerError::TranscriptError)?);

    let point_vec = L::point_to_vec(point.clone());
    sponge.absorb(&point_vec);

    // left-multiply the matrix by `b` - possibly dishonestly
    let mut v = mat.row_mul(&b);

    if dishonesty == LinearCodeDishonesty::RowLCInside {
        v[0] += F::one();
    }

    sponge.absorb(&v);

    // computing the number of queried columns
    let t = calculate_t::<F>(ck.sec_param(), ck.distance(), ext_mat.m)?;

    let indices = get_indices_from_sponge(ext_mat.m, t, sponge)?;

    let mut queried_columns = Vec::with_capacity(t);
    let mut paths = Vec::with_capacity(t);

    let ext_mat_cols = ext_mat.cols();

    for i in indices {
        queried_columns.push(ext_mat_cols[i].clone());
        paths.push(
            col_tree
                .generate_proof(i)
                .map_err(|_| NaysayerError::TranscriptError)?,
        );
    }

    match dishonesty {
        LinearCodeDishonesty::Column => {
            queried_columns[0][0] = F::one();
        }
        LinearCodeDishonesty::MerklePath => {
            paths[0].auth_path[0] = paths[0].auth_path[1].clone();
        }
        LinearCodeDishonesty::MerkleLeafIndex(j) => {
            paths[j].leaf_index += 1;
        }
        LinearCodeDishonesty::RowLCOutside => {
            v[0] += F::one();
        }
        _ => {}
    };

    let opening = LinCodePCProofSingle {
        paths,
        v,
        columns: queried_columns,
    };

    Ok(vec![LinCodePCProof {
        opening,
        well_formedness: None,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_naysay() {
        let rng = &mut test_rng();
        let num_vars = 10;

        let mut pp = TestMLLigero::<Fr>::setup(1, Some(num_vars), rng).unwrap();
        pp.set_well_formedness(false);

        let (ck, vk) = TestMLLigero::<Fr>::trim(&pp, 0, 0, None).unwrap();

        let rand_chacha = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
        let labeled_poly = LabeledPolynomial::new(
            "test".to_string(),
            rand_poly(1, Some(num_vars), rand_chacha),
            Some(num_vars),
            Some(num_vars),
        );

        let test_sponge = test_sponge::<Fr>();
        let (coms, com_states) =
            TestMLLigero::<Fr>::commit(&ck, &[labeled_poly.clone()], None).unwrap();

        let point = rand_point(Some(num_vars), rand_chacha);

        let value = labeled_poly.evaluate(&point);

        // The only arguments to test_naysay_aux we intend to change are the
        // dishonesty and the expected type of naysayer proof returned
        let test_naysay_with = |dishonesty, expected_naysayer_proof| {
            let new_value = if dishonesty != LinearCodeDishonesty::Evaluation {
                value
            } else {
                value + Fr::ONE
            };

            let proof = dishonest_open_lcpcs::<
                MultilinearLigero<_, _, SparseMultilinearExtension<Fr>, _>,
                _,
                _,
                _,
                _,
            >(
                &ck,
                &coms[0],
                &com_states[0],
                &point,
                &mut test_sponge.clone(),
                dishonesty,
            );

            test_naysay_aux::<Fr, SparseMultilinearExtension<Fr>, TestMLLigero<Fr>>(
                &vk,
                &coms,
                &point,
                [new_value],
                &mut test_sponge.clone(),
                proof.unwrap(),
                expected_naysayer_proof,
            );
        };

        /***************** Case 1 *****************/
        // Honest proof verifies and is not naysaid
        test_naysay_with(LinearCodeDishonesty::None, None);

        /***************** Case 2 *****************/
        // Sponge produces different column indices than those in the proof
        test_naysay_with(
            LinearCodeDishonesty::RowLCOutside,
            Some(LinearCodeNaysayerProof {
                incorrect_proof_index: 0,
                naysayer_proof_single: super::LinearCodeNaysayerProofSingle::PathIndexAssertion(0),
            }),
        );

        /***************** Case 3 *****************/
        // Linear encoding pre-image is tampered with post-proof, leading to an
        // inconsistent sponge
        test_naysay_with(
            LinearCodeDishonesty::RowLCInside,
            Some(LinearCodeNaysayerProof {
                incorrect_proof_index: 0,
                naysayer_proof_single: LinearCodeNaysayerProofSingle::ColumnInnerProductAssertion(
                    0,
                ),
            }),
        );

        /***************** Case 4 *****************/
        // Column index is correct, but column values are not
        test_naysay_with(
            LinearCodeDishonesty::Column,
            Some(LinearCodeNaysayerProof {
                incorrect_proof_index: 0,
                naysayer_proof_single: LinearCodeNaysayerProofSingle::MerklePathAssertion(0),
            }),
        );

        /***************** Case 5 *****************/
        // Merkle path proof is tampered with
        test_naysay_with(
            LinearCodeDishonesty::MerklePath,
            Some(LinearCodeNaysayerProof {
                incorrect_proof_index: 0,
                naysayer_proof_single: LinearCodeNaysayerProofSingle::MerklePathAssertion(0),
            }),
        );

        /***************** Case 6 *****************/
        // Merkle path index is manually changed
        test_naysay_with(
            LinearCodeDishonesty::MerkleLeafIndex(17),
            Some(LinearCodeNaysayerProof {
                incorrect_proof_index: 0,
                naysayer_proof_single: LinearCodeNaysayerProofSingle::PathIndexAssertion(17),
            }),
        );

        /***************** Case 7 *****************/
        // Claimed evaluation is incorrect
        test_naysay_with(
            LinearCodeDishonesty::Evaluation,
            Some(LinearCodeNaysayerProof {
                incorrect_proof_index: 0,
                naysayer_proof_single: LinearCodeNaysayerProofSingle::EvaluationAssertion,
            }),
        );

        /***************** Case 8 *****************/
        // Verifier returns false when the proof is correct
        let possible_naysayer_proofs = vec![
            LinearCodeNaysayerProofSingle::PathIndexAssertion(0),
            LinearCodeNaysayerProofSingle::ColumnInnerProductAssertion(0),
            LinearCodeNaysayerProofSingle::MerklePathAssertion(0),
            LinearCodeNaysayerProofSingle::EvaluationAssertion,
            LinearCodeNaysayerProofSingle::PathIndexAssertion(100),
            LinearCodeNaysayerProofSingle::ColumnInnerProductAssertion(100),
            LinearCodeNaysayerProofSingle::MerklePathAssertion(100),
        ];

        let possible_naysayer_proofs = possible_naysayer_proofs
            .into_iter()
            .map(|naysayer_proof_single| LinearCodeNaysayerProof {
                incorrect_proof_index: 0,
                naysayer_proof_single,
            })
            .collect::<Vec<_>>();

        test_invalid_naysayer_proofs::<Fr, SparseMultilinearExtension<Fr>, TestMLLigero<Fr>>(
            &vk,
            &ck,
            [&labeled_poly.clone()],
            &coms,
            &com_states,
            &point,
            &mut test_sponge.clone(),
            possible_naysayer_proofs,
        );
    }
}
