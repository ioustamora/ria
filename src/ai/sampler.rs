use rand::prelude::*;

/// Logits sampling strategy
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    Greedy,
    #[cfg(feature = "greedy_decode")]
    TopK { k: usize },
    #[cfg(feature = "greedy_decode")]
    TopP { p: f32 },
}

/// Sampler configuration
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub strategy: SamplingStrategy,
}

impl Default for SamplerConfig { fn default() -> Self { Self { temperature: 0.8, strategy: SamplingStrategy::Greedy } } }

/// Simple sampler applying temperature + strategy to logits (placeholder implementation)
pub struct LogitsSampler {
    cfg: SamplerConfig,
    rng: ThreadRng,
}

impl LogitsSampler {
    pub fn new(cfg: SamplerConfig) -> Self { Self { cfg, rng: thread_rng() } }

    pub fn sample(&mut self, logits: &[f32]) -> Option<usize> {
        if logits.is_empty() { return None; }
        match self.cfg.strategy {
            SamplingStrategy::Greedy => logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).map(|(i,_)| i),
            #[cfg(feature = "greedy_decode")]
            SamplingStrategy::TopK { k } => self.sample_top_k(logits, k.max(1)),
            #[cfg(feature = "greedy_decode")]
            SamplingStrategy::TopP { p } => self.sample_top_p(logits, p),
        }
    }

    #[cfg(feature = "greedy_decode")]
    fn sample_top_k(&mut self, logits: &[f32], k: usize) -> Option<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|a,b| logits[*b].total_cmp(&logits[*a]));
        let k = k.min(idx.len());
        let slice = &idx[..k];
        self.weighted_choice(logits, slice)
    }

    #[cfg(feature = "greedy_decode")]
    fn sample_top_p(&mut self, logits: &[f32], p: f32) -> Option<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|a,b| logits[*b].total_cmp(&logits[*a]));
        let mut cum = 0f32;
        let mut selected = Vec::new();
        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l / self.cfg.temperature).exp()).collect();
        let mut ordered: Vec<(usize,f32)> = idx.iter().map(|&i| (i, exp_logits[i])).collect();
        let total: f32 = ordered.iter().map(|(_,v)| *v).sum();
        ordered.iter_mut().for_each(|(_,v)| *v /= total.max(1e-9));
        for (i, prob) in ordered.iter() {
            cum += *prob;
            selected.push(*i);
            if cum >= p { break; }
        }
        self.weighted_choice(&exp_logits, &selected)
    }

    #[cfg(feature = "greedy_decode")]
    fn weighted_choice(&mut self, weights_source: &[f32], indices: &[usize]) -> Option<usize> {
        if indices.is_empty() { return None; }
        let mut cum = 0f32;
        for &i in indices { cum += weights_source[i].max(0.0); }
        let r = self.rng.gen::<f32>() * cum.max(1e-9);
        let mut run = 0f32;
        for &i in indices { run += weights_source[i].max(0.0); if run >= r { return Some(i); } }
        Some(*indices.last().unwrap())
    }
}
