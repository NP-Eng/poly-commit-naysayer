use ark_bn254::Fr;
use ark_crypto_primitives::{
    crh::{sha256::digest::Digest, CRHScheme},
    sponge::{
        poseidon::{PoseidonConfig, PoseidonSponge},
        CryptographicSponge,
    },
};
use ark_ff::{PrimeField, UniformRand};
use ark_pcs_bench_templates::{criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup, Criterion};
use ark_poly::{
    univariate::DensePolynomial as DenseUnivariatePoly, DenseMultilinearExtension,
    DenseUVPolynomial, MultilinearExtension, Polynomial, SparseMultilinearExtension,
};
use ark_poly_commit::{
    linear_codes::{LigeroPCParams, LinCodePCProof, LinCodeParametersInfo, MultilinearLigero},
    FieldToBytesColHasher, TestMLLigero, TestMerkleTreeParams,
};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::test_rng;
use blake2::Blake2s256;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha20Rng,
};

use ark_crypto_primitives::merkle_tree::Config;
use ark_crypto_primitives::sponge::Absorb;
use ark_poly_commit::{to_bytes, LabeledCommitment, LabeledPolynomial, PolynomialCommitment};
use core::time::Duration;
use std::{borrow::Borrow, fmt::format, marker::PhantomData, time::Instant};

use ark_poly_naysayer::{
    linear_codes::tests::{dishonest_open_lcpcs, LinearCodeDishonesty},
    linear_codes::LinearCodeNaysayerProof,
    PCSNaysayer,
};

type LigeroParams<F> =
    LigeroPCParams<F, TestMerkleTreeParams, FieldToBytesColHasher<F, Blake2s256>>;

pub fn bench_naysay_ligero_ml(num_vars: usize) {

    let setup_bench = |dishonesty: LinearCodeDishonesty| {
        let mut pp = TestMLLigero::<Fr>::setup(num_vars, Some(num_vars), &mut test_rng()).unwrap();
        pp.set_well_formedness(false);
        let (ck, vk) = TestMLLigero::<Fr>::trim(&pp, 0, 0, None).unwrap();

        let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();

        let mut test_sponge = ark_poly_commit::test_sponge::<Fr>();

        let poly = SparseMultilinearExtension::rand(num_vars, rng);
        let point = (0..num_vars).map(|_| Fr::rand(rng)).collect::<Vec<Fr>>();
        let labeled_poly = LabeledPolynomial::new("test".to_string(), poly, None, None);

        let (coms, com_states) =
            TestMLLigero::<Fr>::commit(&ck, [&labeled_poly], Some(rng)).unwrap();
        let claimed_eval = labeled_poly.evaluate(&point);
        let dishonest_proof = dishonest_open_lcpcs::<
            MultilinearLigero<
                Fr,
                TestMerkleTreeParams,
                SparseMultilinearExtension<Fr>,
                FieldToBytesColHasher<Fr, Blake2s256>,
            >,
            Fr,
            _,
            _,
            _,
        >(
            &ck,
            &coms[0],
            &com_states[0],
            &point,
            &mut test_sponge.clone(),
            dishonesty.clone(),
        )
        .unwrap();

        let new_claimed_eval = if dishonesty != LinearCodeDishonesty::Evaluation {[claimed_eval]} else {[claimed_eval + Fr::from(1)]};

        let naysayer_proof = TestMLLigero::naysay(
            &vk,
            &coms,
            &point,
            new_claimed_eval,
            &dishonest_proof,
            &mut test_sponge.clone(),
            None,
        )
        .unwrap();
        (dishonest_proof, naysayer_proof, vk, coms, point, new_claimed_eval)
    };

    let mut run_honest_benchmark = |id: &str| {
        let mut c = Criterion::default();
        let mut group = c.benchmark_group("Ligero ML");
        group.bench_function(id, |b| {
            b.iter_batched(
                || {
                    setup_bench(LinearCodeDishonesty::None)
                },
                |(dishonest_proof, _, vk, coms, point, claimed_eval)| {
                    TestMLLigero::<Fr>::check(
                        &vk,
                        &coms,
                        &point,
                        claimed_eval,
                        &dishonest_proof,
                        &mut ark_poly_commit::test_sponge::<Fr>(),
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    };

    let mut run_dishonest_benchmark = |id: &str, dishonesty: LinearCodeDishonesty| {
        let mut c = Criterion::default();
        let mut group = c.benchmark_group("Naysayer Ligero ML");
        group.bench_function(id, |b| {
            b.iter_batched(
                || {
                    setup_bench(dishonesty.clone())
                },
                |(dishonest_proof, naysayer_proof, vk, coms, point, claimed_eval)| {
                    TestMLLigero::<Fr>::verify_naysay(
                        &vk,
                        &coms,
                        &point,
                        claimed_eval,
                        &dishonest_proof,
                        &naysayer_proof.unwrap(),
                        &mut ark_poly_commit::test_sponge::<Fr>(),
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    };

    run_dishonest_benchmark(format!("ColumnInnerProductAssertion-{}", &num_vars).as_str(), LinearCodeDishonesty::RowLCInside);
    run_dishonest_benchmark(format!("MerklePathAssertion-{}", &num_vars).as_str(), LinearCodeDishonesty::MerklePath);
    run_dishonest_benchmark(format!("PathIndexAssertion-{}", &num_vars).as_str(), LinearCodeDishonesty::MerkleLeafIndex(0));
    run_dishonest_benchmark(format!("Evaluation-{}", &num_vars).as_str(), LinearCodeDishonesty::Evaluation);

    run_honest_benchmark("VerifyBenchmark");


}

fn bench_naysay_ligero_ml_main() {
    (12..=22).step_by(2).for_each(bench_naysay_ligero_ml);
}

criterion_main!(bench_naysay_ligero_ml_main);
