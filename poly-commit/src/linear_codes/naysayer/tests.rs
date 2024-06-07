use core::borrow::Borrow;

use crate::linear_codes::utils::{calculate_t, get_indices_from_sponge};
use crate::linear_codes::{
    create_merkle_tree, LPCPArray, LinCodePCCommitment, LinCodePCCommitmentState, LinCodePCProof,
    LinCodePCProofSingle, LinCodeParametersInfo, LinearCodePCS, LinearEncode,
};
use crate::utils::test_sponge;
use crate::Error;
use crate::{
    linear_codes::{LigeroPCParams, MultilinearLigero, PolynomialCommitment},
    LabeledPolynomial,
};
use crate::{to_bytes, LabeledCommitment};
use ark_bls12_377::Fr;
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_crypto_primitives::{
    crh::{sha256::Sha256, CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{ByteDigestConverter, Config},
};
use ark_ff::{Field, PrimeField};
use ark_poly::evaluations::multivariate::{MultilinearExtension, SparseMultilinearExtension};
use ark_poly::Polynomial;
use ark_std::test_rng;
use blake2::Blake2s256;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

use ark_pcs_bench_templates::{FieldToBytesColHasher, LeafIdentityHasher};

use super::LinearCodeNaysayerProof;

type LeafH = LeafIdentityHasher;
type CompressH = Sha256;
type ColHasher<F, D> = FieldToBytesColHasher<F, D>;

struct MerkleTreeParams;

impl Config for MerkleTreeParams {
    type Leaf = Vec<u8>;

    type LeafDigest = <LeafH as CRHScheme>::Output;
    type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
    type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

    type LeafHash = LeafH;
    type TwoToOneHash = CompressH;
}

type MTConfig = MerkleTreeParams;

type LigeroPCS<F> = LinearCodePCS<
    MultilinearLigero<F, MTConfig, SparseMultilinearExtension<F>, ColHasher<F, Blake2s256>>,
    F,
    SparseMultilinearExtension<F>,
    MTConfig,
    ColHasher<F, Blake2s256>,
>;

// Types of dishonesty that can be introduce in a LinearCodePCS proof
#[derive(Eq, PartialEq)]
enum LinearCodeDishonesty {
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
    // Modify one of the merkle path proofs, leading to its incorrect
    //  verification
    MerklePath,
    // Claim an evaluation y different from the actual one f(x)
    Evaluation,
}

// Auxiliary functions for testying naysayer functionality
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
    // Convenience function that generates a possibly dishonest LinearCodePCS
    // proof based on the `dishonesty` argument, checks that
    // LinearCodePCS::check outputs the expected result (typically rejection)
    // and also verifies that naysay detects the planted dishonesty and
    // naysay_verify accepts the naysayer proof
    fn test_naysay_aux(
        ck: &L::LinCodePCParams,
        vk: &L::LinCodePCParams,
        com: &LabeledCommitment<LinCodePCCommitment<C>>,
        com_state: &LinCodePCCommitmentState<F, H>,
        point: &P::Point,
        value: F,
        sponge: &mut impl CryptographicSponge,
        dishonesty: LinearCodeDishonesty,
        expected_naysayer_proof: LinearCodeNaysayerProof,
    ) {
        let value = if dishonesty == LinearCodeDishonesty::Evaluation {
            value + F::one()
        } else {
            value
        };

        // Generating possibly dishonest proof
        let proof =
            Self::dishonest_open(ck, com, com_state, point, &mut sponge.clone(), dishonesty)
                .unwrap();

        // TODO This cumbersome block is due to the current inconsistent
        // behaviour of LinearCodePCS::check, which can return Err when
        // there is a genuine runtime error during verification OR when no
        // runtime errors occur but the proof is rejected; and, in the
        // latter case, sometimes Ok(false) is returned instead. The block
        // below can be made cleaner once open is made consistent.
        let result = Self::check(vk, [com], point, [value], &proof, &mut sponge.clone(), None);
        if expected_naysayer_proof == LinearCodeNaysayerProof::Aye {
            assert!(result.unwrap());
        } else {
            assert!(result.is_err() || !result.unwrap());
        }

        // Produce a naysayer proof from the given PCS proof
        let naysayer_proof =
            Self::naysay(vk, [com], point, [value], &proof, &mut sponge.clone(), None).unwrap();

        assert_eq!(naysayer_proof, expected_naysayer_proof);

        // Verify the naysayer proof
        assert_eq!(
            expected_naysayer_proof == LinearCodeNaysayerProof::Aye,
            !Self::naysayer_verify(
                vk,
                [com],
                point,
                [value],
                &proof,
                naysayer_proof,
                &mut sponge.clone(),
                None
            )
            .unwrap()
        );
    }

    // Generates a LinearCodePCS proof introducing a dishonesty
    fn dishonest_open(
        ck: &L::LinCodePCParams,
        com: &LabeledCommitment<LinCodePCCommitment<C>>,
        com_state: &LinCodePCCommitmentState<F, H>,
        point: &P::Point,
        sponge: &mut impl CryptographicSponge,
        dishonesty: LinearCodeDishonesty,
    ) -> Result<LPCPArray<F, C>, Error> {
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
        let mut col_hashes: Vec<C::Leaf> =
            col_hashes.clone().into_iter().map(|h| h.into()).collect(); // TODO cfg_inter

        let col_tree = create_merkle_tree::<C>(
            &mut col_hashes,
            ck.leaf_hash_param(),
            ck.two_to_one_hash_param(),
        )?;

        // 3. Generate vector `b` to left-multiply the matrix.
        let (_, b) = L::tensor(point, n_cols, n_rows);

        sponge.absorb(&to_bytes!(&commitment.root).map_err(|_| Error::TranscriptError)?);

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
                    .map_err(|_| Error::TranscriptError)?,
            );
        }

        match dishonesty {
            LinearCodeDishonesty::Column => {
                queried_columns[0][0] = F::one();
            }
            LinearCodeDishonesty::MerklePath => {
                paths[0].auth_path[0] = paths[0].auth_path[1].clone();
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
}

#[test]
fn test_naysay() {
    let mut rng = &mut test_rng();
    let num_vars = 10;

    let leaf_hash_param = <LeafH as CRHScheme>::setup(&mut rng).unwrap();
    let two_to_one_hash_param = <CompressH as TwoToOneCRHScheme>::setup(&mut rng)
        .unwrap()
        .clone();
    let col_hash_params = <ColHasher<Fr, Blake2s256> as CRHScheme>::setup(&mut rng).unwrap();

    // We assume well-formedness is always false, i.e. that the queried point is
    // random
    let check_well_formedness = false;

    let pp: LigeroPCParams<Fr, MTConfig, ColHasher<Fr, Blake2s256>> = LigeroPCParams::new(
        128,
        4,
        check_well_formedness,
        leaf_hash_param,
        two_to_one_hash_param,
        col_hash_params,
    );

    let (ck, vk) = LigeroPCS::<Fr>::trim(&pp, 0, 0, None).unwrap();

    let rand_chacha = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    let labeled_poly = LabeledPolynomial::new(
        "test".to_string(),
        rand_poly(1, Some(num_vars), rand_chacha),
        Some(num_vars),
        Some(num_vars),
    );

    let test_sponge = test_sponge::<Fr>();
    let (c, rands) = LigeroPCS::<Fr>::commit(&ck, &[labeled_poly.clone()], None).unwrap();

    let point = rand_point(Some(num_vars), rand_chacha);

    let value = labeled_poly.evaluate(&point);

    // The only arguments to test_naysay_aux we intend to change are the
    // dishonesty and the expected type of naysayer proof returned
    let test_naysay_with = |dishonesty, expected_naysayer_proof| {
        LigeroPCS::<Fr>::test_naysay_aux(
            &ck,
            &vk,
            &c[0],
            &rands[0],
            &point,
            value,
            &mut test_sponge.clone(),
            dishonesty,
            expected_naysayer_proof,
        );
    };

    /***************** Case 1 *****************/
    // Honest proof verifies and is not naysaid
    test_naysay_with(LinearCodeDishonesty::None, LinearCodeNaysayerProof::Aye);

    /***************** Case 2 *****************/
    // Sponge produces different column indices than those in the proof
    test_naysay_with(
        LinearCodeDishonesty::RowLCOutside,
        LinearCodeNaysayerProof::PathIndexLie(0),
    );

    /***************** Case 3 *****************/
    // Linear encoding pre-image is tampered with post-proof, leading to an
    // inconsistent sponge
    test_naysay_with(
        LinearCodeDishonesty::RowLCInside,
        LinearCodeNaysayerProof::ColumnInnerProductLie(0),
    );

    /***************** Case 4 *****************/
    // Column index is correct, but column values are not
    test_naysay_with(
        LinearCodeDishonesty::Column,
        LinearCodeNaysayerProof::MerklePathLie(0),
    );

    /***************** Case 5 *****************/
    // Merkle path proof is tampered with
    test_naysay_with(
        LinearCodeDishonesty::MerklePath,
        LinearCodeNaysayerProof::MerklePathLie(0),
    );

    /***************** Case 6 *****************/
    // Claimed evaluation is incorrect
    test_naysay_with(
        LinearCodeDishonesty::Evaluation,
        LinearCodeNaysayerProof::EvaluationLie,
    );
}

fn rand_point<F: Field>(num_vars: Option<usize>, rng: &mut ChaCha20Rng) -> Vec<F> {
    match num_vars {
        Some(n) => (0..n).map(|_| F::rand(rng)).collect(),
        None => unimplemented!(), // should not happen!
    }
}

fn rand_poly<Fr: PrimeField>(
    _: usize,
    num_vars: Option<usize>,
    rng: &mut ChaCha20Rng,
) -> SparseMultilinearExtension<Fr> {
    match num_vars {
        Some(n) => SparseMultilinearExtension::rand(n, rng),
        None => unimplemented!(), // should not happen in ML case!
    }
}
