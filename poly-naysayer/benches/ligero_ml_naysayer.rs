use ark_bn254::Fr;
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::UniformRand;
use ark_pcs_bench_templates::{criterion_main, BatchSize, Criterion};
use ark_poly::{MultilinearExtension, SparseMultilinearExtension};
use ark_poly_commit::{
    linear_codes::{LinCodeParametersInfo, MultilinearLigero},
    FieldToBytesColHasher, LabeledPolynomial, PolynomialCommitment, TestMLLigero,
    TestMerkleTreeParams,
};
use ark_poly_naysayer::{
    linear_codes::tests::{dishonest_open_lcpcs, LinearCodeDishonesty},
    PCSNaysayer,
};

use ark_std::test_rng;
use blake2::Blake2s256;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

use tiny_keccak::{Hasher, Keccak};

#[derive(Clone)]
struct KeccakSponge {
    pub(crate) state: Vec<u8>,
}

impl CryptographicSponge for KeccakSponge {
    type Config = ();

    fn new(_params: &Self::Config) -> Self {
        KeccakSponge { state: vec![] }
    }

    fn absorb(&mut self, input: &impl Absorb) {
        let mut input_bytes = vec![];
        input.to_sponge_bytes(&mut input_bytes);
        self.state.extend_from_slice(&input_bytes);
    }

    fn squeeze_bytes(&mut self, num_bytes: usize) -> Vec<u8> {
        let mut keccak = Keccak::v256();
        let mut output = vec![0u8; num_bytes];
        keccak.update(&self.state);
        keccak.finalize(&mut output);
        self.state = output.clone();
        output
    }

    fn squeeze_bits(&mut self, num_bits: usize) -> Vec<bool> {
        let num_bytes = (num_bits + 7) / 8;
        let tmp = self.squeeze_bytes(num_bytes);
        let dest = tmp
            .iter()
            .flat_map(|byte| (0..8u32).rev().map(move |i| (byte >> i) & 1 == 1))
            .collect::<Vec<_>>();
        dest[..num_bits].to_vec()
    }
}

fn test_sponge() -> KeccakSponge {
    KeccakSponge::new(&())
}

pub fn bench_naysay_ligero_ml(num_vars: usize) {
    let setup_bench = |dishonesty: LinearCodeDishonesty| {
        let mut pp = TestMLLigero::<Fr>::setup(num_vars, Some(num_vars), &mut test_rng()).unwrap();
        pp.set_well_formedness(false);
        let (ck, vk) = TestMLLigero::<Fr>::trim(&pp, 0, 0, None).unwrap();

        let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();

        let test_sponge = test_sponge();

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

        let new_claimed_eval = if dishonesty != LinearCodeDishonesty::Evaluation {
            [claimed_eval]
        } else {
            [claimed_eval + Fr::from(1)]
        };

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
        (
            dishonest_proof,
            naysayer_proof,
            vk,
            coms,
            point,
            new_claimed_eval,
        )
    };

    let run_honest_benchmark = |id: &str| {
        let mut c = Criterion::default();
        let mut group = c.benchmark_group("Ligero ML");
        group.bench_function(id, |b| {
            b.iter_batched(
                || setup_bench(LinearCodeDishonesty::None),
                |(dishonest_proof, _, vk, coms, point, claimed_eval)| {
                    TestMLLigero::<Fr>::check(
                        &vk,
                        &coms,
                        &point,
                        claimed_eval,
                        &dishonest_proof,
                        &mut test_sponge(),
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    };

    let run_dishonest_benchmark = |id: &str, dishonesty: LinearCodeDishonesty| {
        let mut c = Criterion::default();
        let mut group = c.benchmark_group("Naysayer Ligero ML");
        group.bench_function(id, |b| {
            b.iter_batched(
                || setup_bench(dishonesty.clone()),
                |(dishonest_proof, naysayer_proof, vk, coms, point, claimed_eval)| {
                    TestMLLigero::<Fr>::verify_naysay(
                        &vk,
                        &coms,
                        &point,
                        claimed_eval,
                        &dishonest_proof,
                        &naysayer_proof.unwrap(),
                        &mut test_sponge(),
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    };

    run_dishonest_benchmark(
        format!("ColumnInnerProductAssertion-{}", &num_vars).as_str(),
        LinearCodeDishonesty::RowLCInside,
    );
    run_dishonest_benchmark(
        format!("MerklePathAssertion-{}", &num_vars).as_str(),
        LinearCodeDishonesty::MerklePath,
    );
    run_dishonest_benchmark(
        format!("PathIndexAssertion-{}", &num_vars).as_str(),
        LinearCodeDishonesty::MerkleLeafIndex(0),
    );
    run_dishonest_benchmark(
        format!("Evaluation-{}", &num_vars).as_str(),
        LinearCodeDishonesty::Evaluation,
    );

    run_honest_benchmark("VerifyBenchmark");
}

fn bench_naysay_ligero_ml_main() {
    (12..=22).step_by(2).for_each(bench_naysay_ligero_ml);
}

criterion_main!(bench_naysay_ligero_ml_main);
