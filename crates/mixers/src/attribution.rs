//! Resident, independent-leaf component fixtures for backward cost attribution.
//! These call the production operators; they are never substituted into a real model step.
use crate::{Mixer, causal_conv, chunk, hola, profile::Unprofiled};
use burn::{
    prelude::*,
    tensor::{
        Distribution,
        activation::{sigmoid, silu},
    },
};

#[derive(Clone, Copy, Debug)]
pub enum Component {
    Projections,
    Convolutions,
    NormsGates,
    Scan,
    Hola,
    Output,
    Masks,
}
impl Component {
    pub const ALL: [Self; 7] = [
        Self::Projections,
        Self::Convolutions,
        Self::NormsGates,
        Self::Scan,
        Self::Hola,
        Self::Output,
        Self::Masks,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Projections => "Merged input projections",
            Self::Convolutions => "Short causal convolutions (q/k/v)",
            Self::NormsGates => "L2 norms and gate activations",
            Self::Scan => "Fused KronLSTM scan",
            Self::Hola => "HOLA RMSNorms and fused read",
            Self::Output => "Mixer output RMSNorm, gate and projection",
            Self::Masks => "Mixer masks and document IDs",
        }
    }
}

pub struct Fixture {
    input: Tensor<3>,
    projected: [Tensor<3>; 7],
    raw: [Tensor<4>; 3],
    scan: chunk::ScanInput,
    mixed: Tensor<4>,
    exact: Tensor<4>,
    scores: Tensor<4>,
    valid: Tensor<2>,
    reset: Tensor<2>,
    documents: Tensor<2>,
}
impl Fixture {
    pub fn new(mixer: &Mixer, batch: usize, tokens: usize, device: &Device) -> Self {
        let (h, k, v) = (mixer.heads, mixer.key_dim, mixer.value_dim);
        let random3 = |width| {
            Tensor::<3>::random(
                [batch, tokens, width],
                Distribution::Uniform(-0.5, 0.5),
                device,
            )
            .require_grad()
        };
        let random4 = |width| {
            Tensor::<4>::random(
                [batch, tokens, h, width],
                Distribution::Uniform(-0.5, 0.5),
                device,
            )
            .require_grad()
        };
        let raw = [random4(k), random4(k), random4(v)];
        let normalized = |x: Tensor<4>| mixer.normalization_backend.l2(x).detach().require_grad();
        Self {
            input: random3(mixer.q.weight.dims()[0]),
            projected: [h * k, h * k, h * v, h * k, h, h, h * v].map(random3),
            scan: chunk::ScanInput {
                q: normalized(raw[0].clone()),
                k: normalized(raw[1].clone()),
                v: normalized(raw[2].clone()),
                alpha: Tensor::full([batch, tokens, h, k], 0.98, device).require_grad(),
                erase: Tensor::full([batch, tokens, h, 1], 0.5, device).require_grad(),
                write: Tensor::full([batch, tokens, h, 1], 0.5, device).require_grad(),
            },
            raw,
            mixed: random4(v),
            exact: random4(v),
            scores: Tensor::random([batch, tokens, h, 1], Distribution::Uniform(0., 1.), device),
            valid: Tensor::ones([batch, tokens], device),
            reset: Tensor::zeros([batch, tokens], device),
            documents: Tensor::zeros([batch, tokens], device),
        }
    }

    /// A fresh graph from resident independent leaves, including all trainable operator inputs.
    /// Multiple outputs are joined by sum reductions (also present in the forward-only timing).
    pub fn forward(&self, mixer: &Mixer, component: Component) -> Tensor<1> {
        let [b, t, _] = self.input.dims();
        let (h, k, v) = (mixer.heads, mixer.key_dim, mixer.value_dim);
        match component {
            Component::Projections => {
                let packed = mixer.project_input(self.input.clone(), &mut Unprofiled);
                let layers = [
                    &mixer.q,
                    &mixer.k,
                    &mixer.v,
                    &mixer.decay,
                    &mixer.erase,
                    &mixer.write,
                    &mixer.output_gate,
                ];
                packed
                    .into_iter()
                    .zip(layers)
                    .map(|(x, layer)| x.unwrap_or_else(|| layer.forward(self.input.clone())).sum())
                    .reduce(|a, b| a + b)
                    .unwrap()
            }
            Component::Convolutions => [&mixer.q_conv, &mixer.k_conv, &mixer.v_conv]
                .into_iter()
                .enumerate()
                .map(|(i, weight)| {
                    causal_conv(
                        self.projected[i].clone(),
                        weight.val(),
                        self.valid.clone(),
                        self.documents.clone(),
                    )
                    .sum()
                })
                .reduce(|a, b| a + b)
                .unwrap(),
            Component::NormsGates => {
                let widths = [k, k, v];
                let raw: [Tensor<4>; 3] = core::array::from_fn(|i| {
                    silu(self.projected[i].clone()).reshape([b, t, h, widths[i]])
                });
                let norms = raw
                    .into_iter()
                    .map(|x| mixer.normalization_backend.l2(x).sum())
                    .reduce(|a, b| a + b)
                    .unwrap();
                let decay = mixer
                    .normalization_backend
                    .decay(
                        self.projected[3].clone().reshape([b, t, h, k]),
                        mixer.decay_bias.val(),
                        mixer.decay_scale.val(),
                    )
                    .sum();
                let erase = sigmoid(self.projected[4].clone()).sum();
                let write = sigmoid(self.projected[5].clone()).sum();
                let cache = mixer.cache.as_ref().expect("attribution requires HOLA");
                let blend = (self.mixed.clone()
                    + self.exact.clone() * sigmoid(cache.gate.val()).reshape([1, 1, h, 1]))
                .sum();
                norms + decay + erase + write + blend
            }
            Component::Scan => chunk::scan_selected(
                mixer.scan_backend,
                mixer.kind,
                self.scan.clone(),
                self.valid.clone(),
                self.reset.clone(),
                mixer.chunk_size,
                None,
            )
            .output
            .flatten(0, 3),
            Component::Hola => {
                let cache = mixer.cache.as_ref().expect("attribution requires HOLA");
                hola::read_selected(
                    mixer.hola_backend,
                    hola::normalize(
                        mixer.hola_backend,
                        mixer.normalization_backend,
                        &cache.q_norm,
                        self.raw[0].clone(),
                    ),
                    hola::normalize(
                        mixer.hola_backend,
                        mixer.normalization_backend,
                        &cache.k_norm,
                        self.raw[1].clone(),
                    ),
                    self.raw[2].clone(),
                    self.scores.clone(),
                    self.valid.clone(),
                    self.documents.clone(),
                    mixer.cache_capacity,
                    mixer.chunk_size,
                    mixer.cache_policy,
                    cache.sink.val(),
                )
                .flatten(0, 3)
            }
            Component::Output => {
                let mixed = mixer.normalization_backend.gated_rms(
                    &mixer.norm,
                    self.mixed.clone(),
                    self.projected[6].clone().reshape([b, t, h, v]),
                );
                mixer
                    .project_output(mixed.reshape([b, t, h * v]))
                    .flatten(0, 2)
            }
            Component::Masks => {
                let documents = (self.reset.clone() * self.valid.clone()).cumsum(1);
                // The IDs have no VJP, but their real forward work belongs in this bucket.
                let input = self.input.clone() * self.valid.clone().unsqueeze_dim::<3>(2);
                let output = self.input.clone() * self.valid.clone().unsqueeze_dim::<3>(2);
                input.sum() + output.sum() + documents.sum()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn isolated_components_build_fresh_graphs_and_input_vjps() {
        let device = Device::ndarray().autodiff();
        let mixer = crate::MixerConfig {
            kind: crate::MixerKind::KronLstm,
            cache_policy: crate::CachePolicy::Surprise,
            d_model: 8,
            heads: 2,
            key_dim: 3,
            value_dim: 5,
            chunk_size: 4,
            cache_capacity: 3,
            projection_backend: crate::ProjectionBackend::Merged,
            ..Default::default()
        }
        .init(&device);
        for batch in [1, 2] {
            let fixture = Fixture::new(&mixer, batch, 7, &device);
            for component in Component::ALL {
                for _ in 0..2 {
                    let output = fixture.forward(&mixer, component);
                    let gradients = output.backward();
                    let input: Tensor<1> = match component {
                        Component::Projections | Component::Masks => {
                            fixture.input.grad(&gradients).unwrap().flatten(0, 2)
                        }
                        Component::Convolutions | Component::NormsGates => {
                            fixture.projected[0].grad(&gradients).unwrap().flatten(0, 2)
                        }
                        Component::Scan => fixture.scan.q.grad(&gradients).unwrap().flatten(0, 3),
                        Component::Hola => fixture.raw[0].grad(&gradients).unwrap().flatten(0, 3),
                        Component::Output => fixture.mixed.grad(&gradients).unwrap().flatten(0, 3),
                    };
                    assert!(
                        input
                            .into_data()
                            .try_to_vec::<f32>()
                            .unwrap()
                            .iter()
                            .all(|x| x.is_finite())
                    );
                }
            }
        }
    }
}
