---
layout: post
title: "Give Shakespeare Infinite Monkeys; Anatomy of Slop"
author: Théo Gigant
use_math: false
category: ai creativity
image: TODO.jpg
---

*Illustration: Fountain (Marcel Duchamp, 1917, photographed by Alfred Stieglitz)*

<!-- TODO: personal hook. Replace this paragraph with a real example of your own (a track, a render, a codebase). Draft version below uses a generic one. -->

Open FL Studio, load the default kick, drop a stock 808 pattern, pick the first preset on the first synth you find, and you will have a beat in ten minutes. It will sound like ten thousand other beats. Nobody calls this "DAW slop", but everybody who has spent time on SoundCloud knows exactly what it sounds like.

The same goes for Unreal Engine scenes with the default sky and the default lighting, for stock-template websites, for photos taken on auto mode of a sunset. None of these tools is bad. They all have a default setting, and the default setting is where taste goes to die.

**"Slop" is not a property of a medium. It is what you get when nobody made any decisions.**

Generative AI did not invent slop. What it changed is the price: the default output of a diffusion model or a language model is surprisingly competent, it costs nothing, and it comes out at a volume no preset pack could ever match. That is a real change, and it deserves a real discussion. But it is a change in the economics of defaults, not proof that the tool can't be used with intent.

This post is written for two groups who don't always talk to each other: people who build with these tools, and artists who are skeptical of them, especially the ones who write a bit of code and know what a good tool feels like. In it I try to take the most common critiques against AI as a creative tool seriously, separate the ones I think are right from the ones I think are confused, and argue for a simple position: **what matters is who makes the decisions, not what executes them.**

---

## The strongest case against

The best version of the argument against AI art that I know of is Ted Chiang's essay [Why A.I. Isn't Going to Make Art](https://www.newyorker.com/culture/the-weekend-essay/why-ai-isnt-going-to-make-art) (The New Yorker, 2024). It's worth reading in full, but the core idea fits in one sentence: art is something that results from making a lot of choices.

A novelist writing ten thousand words makes, roughly, ten thousand choices. A painter makes choices with every brushstroke. When you type a short prompt into an image generator, you made a handful of choices, and the model fills in everything else, by averaging over what it has seen. The result, Chiang argues, can't carry much intention because there was very little intention put in.

I think this is mostly right, and I want to be clear about that before disagreeing with it. **Default prompting really does delegate almost every decision to the model.** Lighting, composition, palette, the expression on a face, the rhythm of a sentence: if you didn't specify it, you didn't choose it. The model chose the most likely option, which is, by construction, the most average one. This is exactly why default AI output looks the way it does. It's the statistical center of everything.

Where I disagree is with the jump from "default prompting delegates choices" to "the tool can't produce art". That jump assumes that a short text prompt is the only way to use these models. It isn't.

## Friction moves, it doesn't disappear

Every creative tool removes some friction and leaves some. A camera removes the need to render light by hand; it doesn't remove the need to decide where to stand. A synthesizer removes the need to build an oscillator out of transistors; it doesn't remove the need to decide what the sound should be. A compiler removes the need to write assembly; it doesn't remove the need to design the program.

This is the old idea behind Douglas Engelbart's [Augmenting Human Intellect](https://www.dougengelbart.org/content/view/138) (1962): tools should let people work on problems at a higher level of abstraction, not do the work for them. The effort doesn't vanish. It moves up a level.

Generative models fit this pattern, with one important difference that the music and photography analogies tend to gloss over:

- **Traditional tools automate execution.** The synth generates the waveform, the camera sensor captures the photons, but every decision about *what* to generate or capture is still made by a person.
- **Generative models automate decisions.** The model decides the lighting, the composition, the word choice. That's what makes default output so easy, and so empty.

So the question becomes: can you take the decisions back?

In practice, yes, and the tools for doing it already exist. In image generation, there is inpainting (regenerate only the region you're not happy with), ControlNet (fix the pose, the depth map, the edges, and let the model fill only the texture), fixed seeds and parameter sweeps (change one thing at a time and compare), fine-tuning on your own material, and custom pipelines that chain all of these. In writing and code, it looks like writing the structure yourself and delegating only the parts you've already decided on, then rejecting most of what comes back.

Each of these hands decisions back to the human. A heavily controlled pipeline is not "typing a prompt". The person running it makes hundreds of choices, they're just made at a different level than brushstrokes. That's the same shift a photographer made relative to a painter.

<!-- TODO: concrete example from your own work. What did you automate, what did you refuse to automate, and why? E.g. "I don't care about writing CUDA boilerplate, I care about the architecture" or a music example. This paragraph is the most important one in the post. -->

The rule I try to follow is simple: **automate the execution in domains you don't care to master, and keep the decisions in the domains you do.** If I don't care how a function is spelled but care a lot about how the system is structured, delegating the spelling is fine. If I let the model also decide the structure, the result is no longer mine, and it will probably look like everybody else's.

## Curation is a creative act

The other common objection is that even a controlled pipeline turns the creator into a *curator*: you're picking from outputs rather than making them. That's true. I just don't think it's the gotcha it's meant to be.

In 1917, Marcel Duchamp submitted a store-bought urinal, signed "R. Mutt", to an art exhibition. He didn't make the object. He chose it, framed it, and named it, and that act of selection is now one of the most discussed works of the twentieth century. You don't have to like *Fountain* to accept the point it made: choosing and framing can be the work.

Curation is also most of what a film editor, a DJ, a magazine editor or an art director does. Nobody thinks these jobs are trivial, precisely because choosing well is hard. What makes curation creative is not the absence of making; it's the presence of constraints, criteria, and a point of view.

<!-- TODO: expand your 100k-outputs performance idea here. Suggested framing below. -->

Here's a thought experiment I keep coming back to. Imagine generating a hundred thousand outputs from a model, and then evaluating, weighting and selecting from them by hand, according to criteria you've defined and refined over the course of the process. The model produced every individual image. But the final selection, its ordering, and what it says about the distribution it came from are entirely the result of human decisions, made at enormous cost in time and attention. That's not *less* friction than painting. It's a different kind of friction.

The lazy version of curation (generate four images, post the least weird one) is slop. The demanding version isn't. Same tool, different amount of intent.

## We've had this argument before

Every time a tool has made some artifact much cheaper to produce, a version of this debate has happened.

In his review of the [Salon of 1859](https://gallowayexploringart.wordpress.com/wp-content/uploads/2014/08/baudelaire_the-modern-public-photography.pdf), Charles Baudelaire described photography as "the refuge of every would-be painter, every painter too ill-endowed or too lazy to complete his studies", and warned that if it was allowed to supplement art, it would soon supplant or corrupt it altogether. In 1935, Walter Benjamin's [The Work of Art in the Age of Mechanical Reproduction](https://www.marxists.org/reference/subject/philosophy/works/ge/benjamin.htm) took the question seriously instead of dismissing it, asking what happens to the "aura" of a work when it can be reproduced infinitely.

In music, the same thing happened with samplers, drum machines, DAWs, software instruments, and Auto-Tune. Each one was accused of killing musicianship. Each one did kill a certain kind of musicianship (fewer people need to learn to engineer tape), and each one also enabled genres that couldn't have existed otherwise.

The recurring pattern is a confusion between **labor** and **vision**. When the labor needed to make an artifact drops, critics reasonably notice that a lot of low-effort artifacts appear, and conclude that the tool itself is the problem. Over time, people learn to tell the tool from the intent, and the critique shifts from "this tool isn't real art" to "this particular work is lazy", which is the right critique.

To be fair, the analogy is not perfect. As discussed above, a camera doesn't decide your composition for you, and a diffusion model does, unless you stop it. So "photography was also criticized" isn't a complete answer by itself. It's a reminder that "it's too easy" has historically been a weak predictor of whether a medium ends up producing art.

## The critiques I take seriously

Not every critique of AI is about creative validity, and some of them are right. I think the conversation gets worse when these are mixed together, so here they are, separately.

### Data, consent and compensation

Most large models were trained on work collected without consent or compensation, and those models now compete in the same markets as the people whose work trained them. That's a real labor problem and an open legal one, and "but the model is only approximating a distribution" doesn't address it. The complaint isn't about the math, it's about who benefits.

I support fair compensation for the people whose work is used, and I'd like to see more models trained on licensed or opt-in data. But this is a question about how the tools are *built and owned*, not about whether a person using one can make something with intent. You can hold both positions at once, and I do.

### "It just copies"

Mostly no, sometimes yes. Diffusion models don't store their training images and retrieve them; they learn to approximate a distribution. But [Carlini et al. (2023)](https://arxiv.org/abs/2301.13188) showed that models *can* memorize and reproduce some training images almost exactly, mostly images that were duplicated many times in the training set. It's rare, but it's not zero, and anyone claiming "models can't copy" is overclaiming.

The subtler issue is *style*. A model fine-tuned on one illustrator's portfolio doesn't copy any single image, but it can reproduce something that used to be impossible to separate from the person: the way they draw. Our legal and ethical frameworks weren't built for this, and I don't think it should be waved away.

### "It looks bad"

This one I'll mostly dismiss, but not for the usual reason. The six-finger hands are mostly gone, and fidelity will keep improving. But that doesn't defeat the critique, it sharpens it: if the output gets more polished and still feels empty, then polish was never what made work good. Detail used to be a signal of effort and intent; now it's free, so it's not a signal of anything. That's an argument *for* my position. When execution is free, intent is the only thing left to judge.

### Skill atrophy

This is the critique I agree with most. If you delegate everything, you lose the ability to do it yourself, and more importantly, you lose the ability to *judge* what the tool gives you. You can't spot a subtle bug in code you could never have written, and you can't tell a good composition from an average one if you've never had to compose.

That's the catch in "automate what you don't care to master": you still need enough skill to evaluate the output, otherwise you're not choosing, you're just accepting. Using AI with intent requires *more* taste and judgment, not less. The people best placed to use these tools well are often the ones who could do without them.

## Who made the decisions?

"Is it AI or not" is the wrong question. It treats a tool as if it were a verdict.

The useful question is the one Chiang points at, just applied more carefully: **who made the decisions?** If the answer is "the model, by default", the result is slop, whether it came from a diffusion model, a preset pack, or a template. If the answer is "a person, deliberately, at whatever level of abstraction they chose to work at", then it's worth judging the work on its merits, like anything else.

Slop is what defaults look like. The fix has never been to ban the tool. It's to make more of the choices yourself.
