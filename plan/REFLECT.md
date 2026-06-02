i am doing ml side research:
neural lm compression with range coding and random seed basis and lora / hira
continuous tokenizer with fsq bsq bytes to latent to bytes, fsq used because like vq but easy
optimizer compiler given model dag and loss, traverse ad tape propose optims to use or some curvature fixing per layer
continous lm that learn to use error without backprop, update kv cache learn corpus reupdate

what is common theme
is it right track to do above or too stupid

None of those ideas sound stupid. In fact, they cluster around a surprisingly coherent research direction.

The common theme I see is:

> **Can we replace discrete, expensive, globally optimized ML systems with continuous, adaptive, locally updateable information-processing systems?**

More concretely, you're repeatedly attacking four assumptions that modern deep learning takes for granted:

1. **Weights are the primary storage medium**

   * Your LM compression work asks: can knowledge be stored as seeds, low-rank updates, codes, or compressed representations rather than billions of explicit parameters?
   * LoRA/HiRA, range coding, random basis generation all point toward "weights as compressed programs."

2. **Tokens are discrete**

   * Continuous tokenizer with FSQ/BSQ is asking:
   * Why should language be represented as vocab indices at all?
   * Can language live in a continuous latent communication channel?
   * This is very similar in spirit to neural codecs and latent language modeling.

3. **Backprop is the only learning mechanism**

   * Learning from error without standard backprop.
   * Local adaptation via KV cache.
   * Online corpus re-updating.
   * This is exploring alternatives to global gradient descent.

4. **Optimization is hand-designed**

   * Optimizer compiler from DAG + loss.
   * Auto-select curvature corrections.
   * Layer-specific update rules.
   * Treat optimization itself as a compilation problem.

Those four ideas all challenge fixed design choices in current ML stacks.

---

The stronger formulation of your research agenda might be:

> "Machine learning systems should learn representations, memory structures, and update rules automatically rather than relying on fixed tokens, fixed weights, fixed optimizers, and fixed backpropagation."

That's actually a pretty deep agenda.

---

What I would be careful about is that some of your ideas are much more grounded than others.

### Strong / promising

#### Continuous tokenizer with FSQ/BSQ

This is close to active research.

People are already exploring:

* latent language models
* neural codecs
* byte-level models
* residual quantization
* VQ alternatives

FSQ especially is attractive because VQ's codebook collapse issues are annoying.

I think this is probably your most publishable-looking direction.

---

#### LM compression + LoRA + coding

Also strong.

The world still wants:

* smaller models
* better storage
* efficient personalization

Compression research never dies.

The question becomes:

> Can a compressed representation be learned directly rather than compressing after training?

That's interesting.

---

#### Optimizer compiler

Potentially very interesting.

Most optimizers are manually designed:

* SGD
* Adam
* Adafactor
* Shampoo
* Muon
* Sophia

An optimizer compiler that inspects graph structure and proposes update rules is unusual.

The risk is complexity explosion.

The opportunity is that modern models contain many substructures that probably want different optimizers.

---

### More speculative

#### Learning without backprop

This is where many researchers accidentally disappear into the wilderness.

The question isn't:

> Can I avoid backprop?

The question is:

> Can I beat or match backprop's sample efficiency?

Backprop is ridiculously hard to beat.

Many alternatives:

* Hebbian learning
* predictive coding
* equilibrium propagation
* feedback alignment

look beautiful theoretically but struggle practically.

So I'd treat this as high-risk research.

---

#### KV-cache learning

Interesting but dangerous.

KV caches are usually:

* transient memory
* inference state

Turning them into learning state starts looking like:

* fast weights
* associative memory
* retrieval systems
* online adaptation

There is definitely research value there.

But you'll need a mechanism preventing catastrophic drift.

---

What I notice most is that all your projects seem driven by a single instinct:

> "Current deep learning systems waste information."

You're trying to remove waste in:

* representation (continuous tokenization)
* storage (compression)
* optimization (optimizer compiler)
* adaptation (KV-cache learning)

That's a legitimate research taste. Many influential researchers have a similar pattern where they keep attacking inefficiencies from different angles.

---

If I were prioritizing:

1. Continuous tokenizer / latent language modeling
2. Compression + LoRA-style factorization
3. Optimizer compiler
4. KV-cache adaptation
5. Backprop alternatives

in roughly that order of expected payoff.

The only thing I'd watch for is becoming too broad. All four ideas could each consume a PhD.

A useful question to ask yourself is:

> "If all these projects succeed, what principle have I demonstrated?"

If you can answer that in one sentence, then you probably have a real research agenda rather than a collection of disconnected experiments. From what you've written, that sentence might be:

> "Learning systems should compress, represent, and update information continuously and adaptively instead of relying on fixed discrete structures."

That's a coherent direction, not a stupid one.
