use crate::linear_codes::LinearCodePCS;
use crate::utils::test_sponge;
use crate::{
    linear_codes::{LigeroPCParams, MultilinearLigero, PolynomialCommitment},
    LabeledPolynomial,
};
use ark_bls12_377::Fr;
use ark_crypto_primitives::{
    crh::{sha256::Sha256, CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{ByteDigestConverter, Config},
    sponge::poseidon::PoseidonSponge,
};
use ark_ff::AdditiveGroup;
use ark_poly::evaluations::multivariate::SparseMultilinearExtension;
use ark_std::test_rng;
use blake2::Blake2s256;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

use ark_pcs_bench_templates::{FieldToBytesColHasher, LeafIdentityHasher};

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
type Sponge<F> = PoseidonSponge<F>;

type LigeroPCS<F> = LinearCodePCS<
    MultilinearLigero<
        F,
        MTConfig,
        Sponge<F>,
        SparseMultilinearExtension<F>,
        ColHasher<F, Blake2s256>,
    >,
    F,
    SparseMultilinearExtension<F>,
    Sponge<F>,
    MTConfig,
    ColHasher<F, Blake2s256>,
>;

#[test]
fn test_naysay_ligero_ml() {
    let mut rng = &mut test_rng();
    let num_vars = 10;
    // just to make sure we have the right degree given the FFT domain for our field
    let leaf_hash_param = <LeafH as CRHScheme>::setup(&mut rng).unwrap();
    let two_to_one_hash_param = <CompressH as TwoToOneCRHScheme>::setup(&mut rng)
        .unwrap()
        .clone();
    let col_hash_params = <ColHasher<Fr, Blake2s256> as CRHScheme>::setup(&mut rng).unwrap();
    // for now, assume well-formedness is always false
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

    let mut proof = LigeroPCS::<Fr>::open(
        &ck,
        &[labeled_poly],
        &c,
        &point,
        &mut (test_sponge.clone()),
        &rands,
        None,
    )
    .unwrap();

    // proof verifies
    assert!(LigeroPCS::<Fr>::check(
        &vk,
        &c,
        &point,
        [value],
        &proof,
        &mut (test_sponge.clone()),
        None
    )
    .unwrap());

    let naysayer_proof = LigeroPCS::<Fr>::naysay(
        &vk,
        &c,
        &point,
        [value],
        &proof,
        &mut (test_sponge.clone()),
        None,
    )
    .unwrap();

    println!("Naysayer proof: {:?}", naysayer_proof);
    assert!(naysayer_proof == LinearCodeNaysayerProof::Aye);

    let original_proof = proof.clone();

    // maliciously modify the proof

    // first try modifying `v`
    proof[0].opening.v = vec![Fr::ZERO; proof[0].opening.v.len()];

    assert!(LigeroPCS::<Fr>::check(
        &vk,
        &c,
        &point,
        [value],
        &proof,
        &mut (test_sponge.clone()),
        None
    )
    .is_err());

    let naysayer_proof = LigeroPCS::<Fr>::naysay(
        &vk,
        &c,
        &point,
        [value],
        &proof,
        &mut (test_sponge.clone()),
        None,
    )
    .unwrap();

    // but naysayer proof is accepted
    assert!(LigeroPCS::<Fr>::naysayer_verify(
        &vk,
        &c,
        &point,
        [value],
        proof,
        naysayer_proof,
        &mut (test_sponge.clone()),
        None
    )
    .unwrap());

    // modify a column
    proof = original_proof;
    proof[0].opening.columns[0] = vec![Fr::ZERO; proof[0].opening.columns[0].len()];

    // proof fails
    assert!(LigeroPCS::<Fr>::check(
        &vk,
        &c,
        &point,
        [value],
        &proof,
        &mut (test_sponge.clone()),
        None
    )
    .is_err());
}
