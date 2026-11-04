---
layout: post
title: "Give Shakespeare Infinite Monkeys; Anatomy of Slop"
author: Théo Gigant
use_math: false
category: ai creativity
image: chimpanzee-typewriter.jpg
---

*Illustration: An AI agent, probably not writing Hamlet (New York Zoological Society, 1906, public domain)*

Here is a 3D render I made in Blender, a professional, open-source 3D suite used in films and games. No AI was involved in this process.

<img src="{{site.baseurl}}/assets/img/no-ai-art.png" width="600" />
<p style="text-align:center;">A handmade, 100% human, AI-free 3D artwork<br></p>

So it isn't slop, is it?

Anyone who has opened Blender recognizes Suzanne, the monkey head that ships with the software as a test mesh, sitting under the default light, seen through the default camera.
It took me around five minutes to figure out how to do this, and I didn't make a single decision that mattered.
It's the 3D equivalent of playing "Come as You Are" on the guitar, or printing `Hello World!` in C.

None of these tools is bad. They all have a default setting, and the default setting is where nobody decided anything.

On the Wikipedia page for AI slop you can find this definition:

> The goal of pig slop is to maximize nutrient intake while minimizing cost; the goal of A.I. slop is to maximize time spent consuming content while minimizing cost.  
> — *Oliver Whang, "Can You Get Rich Quick Off A.I. Slop?". The New York Times Magazine.*

Note that the AI part isn't necessary: any method that creates a large volume of content with minimum effort is producing slop.

**Slop is not a property of a medium. It is what you get when nobody made any decisions.**

Slop preceded generative AI and could outlive it. What AI enabled is cheap and decent-enough slop: the default output of a diffusion model or a language model is surprisingly competent, costs almost nothing, and can be milked at an industrial scale.
That is a real change, and it deserves a real discussion. But it is a change in the consequences of default settings, not proof that the tool cannot be used with intent.

This post is about whether you can make something with intent using these tools, not about how they were trained; the consent and compensation questions are real and deserve their own post.
It is written for two groups who don't always talk to each other: people who already build with these tools, and creatives who are skeptical of them.
To the first group I will say: know what decisions you are not making by using AI; to the second group: you don't have to make *all* the decisions, only the ones that matter for what you're trying to convey.

---

## A thousand choices

Ted Chiang wrote a really good essay about AI and art: [Why A.I. Isn't Going to Make Art](https://www.newyorker.com/culture/the-weekend-essay/why-ai-isnt-going-to-make-art) (The New Yorker, 2024). It's worth reading in full, but the core idea fits in one sentence: art is something that results from making a lot of choices.

A novelist writing ten thousand words makes, roughly, ten thousand choices. A painter makes choices with every brushstroke. When you type a short prompt into an image generator, you made a handful of choices, and the model fills in everything else, with statistically plausible content given the context you gave.
The result, Chiang argues, can't carry much intention because there was very little intention put in.

I agree with this take. **Default prompting really does delegate almost every decision to the model.** Lighting, composition, palette, the expression on a face, the rhythm of a sentence: if you didn't specify it, you didn't choose it.
The model chose the most likely option, which is, by construction, an unsurprising one.
This is exactly why default AI output looks the way it does.
It's the statistical center of everything.

But Chiang also says:
> Is there [an] opportunity to make a vast number of choices using a text-to-image generator? I think the answer is no. [...]
> We can imagine a text-to-image generator that, over the course of many sessions, lets you enter tens of thousands of words into its text box to enable extremely fine-grained control over the image you’re producing; this would be something analogous to Photoshop with a purely textual interface. I’d say that a person could use such a program and still deserve to be called an artist.

Chiang's hypothetical program already exists. Some are specialized image pipelines like [ComfyUI](https://github.com/comfy-org/comfyui), [NodeTool](https://github.com/nodetool-ai/nodetool) or [Slopcannon](https://slop.cc/wiki/); general agent harnesses like [Hermes Agent](https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/creative/creative-hyperframes) are the same idea applied to any medium: a textual interface where you specify what you want, reject what comes back, and re-roll until the output matches what you had in mind. (Transparency aside: I work for the company building one of these, so discount accordingly.)

In the end, it's you making art with AI, not AI making art.
Don't be afraid to explore AI as a tool, but reject the default: think about the thousand choices you did not make, and make a thousand more elsewhere.

---

## Friction moves, it doesn't disappear

Every creative tool removes some friction and leaves some.
A camera removes the need to render light by hand; it doesn't remove the need to think about picture composition.  
A MIDI sequencer removes the need to perform perfectly; it doesn't remove the need to decide what the melody should be.  
A compiler removes the need to write assembly; it doesn't remove the need to design the program.

Generative models fit this pattern, with caveats:

- **Traditional tools automate execution.** The synth generates the waveform, the camera sensor captures the photons, but every decision about *what* to generate or capture is still made by a person (*well*, [or monkey](https://en.wikipedia.org/wiki/Monkey_selfie_copyright_dispute)).
- **Generative models automate decisions.** The model decides the lighting, the composition, the word choice. That's what makes default output so easy, and so empty.

So the question becomes: can you take the decisions back?

In practice, yes, and the tools for doing it already exist.
In image generation, there is inpainting (regenerate only the region you're not happy with), ControlNet (fix the pose, the depth map, the edges, and let the model fill only the texture), fixed seeds and parameter sweeps (change one thing at a time and compare), fine-tuning on your own material, and custom pipelines that chain all of these.
In writing and code, it looks like writing the structure yourself and delegating only the parts you've already decided on, then rejecting, resampling and iterating until you're happy with the result.

Each of these hands decisions back to the human. A heavily controlled pipeline is not "typing a prompt".
The person running it makes thousands of choices; it's just that they're made at a different level than brushstrokes.
That's the same shift a photographer made relative to a painter.

---

## What I hand off, and what I don't

Let me illustrate this with my own process.
I produce electronic music as a hobby, and doing it from start to finish is time consuming. To get tracks that sound decent enough to be played in actual DJ sets (with at least [*one*](https://youtu.be/cOoiwq8NT7M?si=_MoldrDz64x-fGux&t=2620) major recorded success), I had to learn the basics of mixing and mastering.
That skill fades quickly: after a few months without practice, I have to relearn most of it.
So I started delegating mixing and mastering to AI tools.
The alternative would be paying an engineer, which hands off the same decisions but costs more than I'm comfortable spending on a hobby.
What I kept from learning the basics is the ability to hear when a mix is wrong, and that's enough to reject what the tool gives me.
I also use AI for ideation and song-starter loops, replacing a *default* default (browsing Splice for starter samples) with a *less default* one.
I'm not fully happy with this setup yet, and I keep adjusting it. But the decisions I care about, what the track is and how it should feel, are still mine, and the setup is far removed from the defaults.

This blog post is another example.
I used to write everything myself and only asked an LLM to rephrase the occasional sentence.
This time, I delegated the parts I don't care much about.
The prose started as default Opus output, guided by my older posts as a style reference, and I rewrote the parts I didn't like.
I also used agents for the literature review: I started yapping about my ideas, they suggested readings, and I rejected most of them.
I already knew where the argument was going; the model helped me find weak points and tighten the structure.
What makes this post (at least partially) mine is the decisions: what it argues, what it concedes, and what I cut.

Let me suggest this simple rule: **automate the execution in domains you don't care to master, and keep the decisions in the domains you do.**
If I don't care how a function is spelled but care a lot about how the system is structured, delegating the spelling is fine.
If I let the model also decide the structure, the result is no longer mine, and it will probably look like everybody else's.

---

## Learn enough to judge, not enough to perform

There is a catch in "automate what you don't care to master":

**You can only make decisions in the dimensions you can perceive.**
If I don't know much about music, I won't hear that a chord progression is generic.
I'll accept it without noticing that I accepted anything.
In that case the model didn't just execute for me, it quietly made a choice I didn't know existed.
Skill in a medium is both the ability to execute and the vocabulary that lets you *have intentions* in that medium at all.

So the people who say you need to know the craft are partly right. Where I disagree is about *which* part of the craft you need.
You don't need to be able to play the guitar part.
You need to be able to say why this take is worse than that one.
That's the difference between a performer and an editor, and it's a much smaller (and different) investment.

Rick Rubin (the OG vibe-producer), in his book *The Creative Act*, puts a lot of weight on this kind of training: paying attention, taking in great work, developing the ability to notice.
It's not glamorous advice, but it's the honest answer to "how do I use these tools without producing slop": consume a lot of good work in the medium, until you can tell what's off and why.

In other words: **learn enough to judge, not enough to perform.** And where you can't judge, assume the model decided, and be honest about it, at least with yourself.

---

## Curation is a creative act

The other common objection is that even a controlled pipeline turns the creator into a *curator*: you're picking from outputs rather than making them. That's true. I just don't think it's the gotcha it's meant to be.

From Marcel Duchamp's urinal to Daft Punk's heavy sampling: choosing and framing can be the creative work.

> > "Good Artists Copy; Great Artists Steal"
> > 
> > — *Pablo Picasso*
> 
> — *Steve Jobs*


Curation is also most of what a film editor, a DJ, a magazine editor or an art director does.
And these jobs are reputation-heavy, precisely because choosing well is hard.
What makes curation creative is not the absence of making; it's the presence of constraints, criteria, and a point of view.

Film is the clearest case. Films are routinely credited to their director, someone who usually doesn't act, light the set, operate the camera or write the score.
This wasn't always obvious: it took the critics of the [auteur theory](https://en.wikipedia.org/wiki/Auteur_theory) to argue that a film has an author, and that the author is the person whose decisions shape the whole.
Music has its own version, and it's Rick Rubin again: he openly says he isn't a technical musician, and his contribution is the trained ear, knowing what he wants to hear and refusing everything else.
Nobody says the records he produced aren't at least partly his.

The lazy version of curation (generate four images, post the least weird one) is slop. The demanding version isn't. Intent is what sets these apart.

---

## We've had this argument before

Every time a tool has made some artifact much cheaper to produce, a version of this debate has happened.

[Charles Baudelaire](https://www.metmuseum.org/art/collection/search/270956) described the photographic industry as "the refuge of every would-be painter, every painter too ill-endowed or too lazy to complete his studies".

In music, a similar argument was thrown around with every new technology, including samplers, drum machines, DAWs, software instruments, and Auto-Tune.
Each one was accused of killing musicianship.
Each one *did* hurt a certain kind of musicianship (fewer people need to learn to engineer tape), and each one also enabled genres that couldn't have existed otherwise.

The recurring pattern is a confusion between **labor** and **vision**. When the labor needed to make an artifact drops, critics reasonably notice that a lot of low-effort artifacts appear, and conclude that the tool itself is the problem. Over time, people learn to tell the tool from the intent, and the critique shifts from "this tool isn't real art" to "this particular work is lazy", which is the right critique.

To be fair, the analogy is not perfect. As discussed above, a camera doesn't decide your composition for you, and a diffusion model does, unless you stop it. So "photography was also criticized" isn't a complete answer by itself. It's a reminder that "it's too easy" has historically been a weak predictor of whether a medium ends up producing art.

---

## Meaning and provenance

If the decisions are what make a work yours, why do so many people still feel that AI-assisted work is worth less, even when the decisions are clearly human? I think it's because we value works for two different reasons, and we rarely separate them.

- **Meaning**: what the work says, how it's put together, what it makes you feel. This is a direct consequence of the artist's decisions, and it can survive the use of any tool, including AI, if those decisions were made deliberately.
- **Provenance**: the fact that a person spent the hours. The friction you can see, the steps nobody did for them.

A handmade wooden table is a good example of provenance value. An IKEA table holds your plates just as well; the handmade one is more expensive because of the hours and the hands behind it. Plenty of art carries both kinds of value, and for a lot of audiences, visible effort is part of how they relate to a work. It's a signal of care, a shared human cost.

Generative AI doesn't really threaten meaning. It threatens provenance, because it makes effort invisible and unverifiable. SHL0MS' [*Inferior Image*](https://x.com/SHL0MS/status/2054280631807316329?s=20) is a clean demonstration: he posted a real Monet, claimed it was AI-generated, and collected hundreds of confident critiques.
The same pixels went from "obviously a masterpiece" to "obviously slop" depending only on what people believed about how they were made.

I don't think this is irrational, and I don't think it goes away. My guess is that it will become its own market, the way it already did for furniture, bread, or clothes.
I would not be surprised to see "artisanal" code (or ["digitally handmade"](https://www.instagram.com/reel/DeeoOvFvcEQ/?mdxt=MXQ2cG56dGk0MnRmMg==)) in a few years: software written entirely by humans, sold at a premium, with the provenance as the selling point.
That's fine. It just means we should be clear about which value we're talking about.
"This took no effort" is a statement about provenance.
"This says nothing" is a statement about meaning.
Only the second one is a critique of the work.

---

## Artifacts of contemporary slop will age into style

Nowadays it would be an artistic choice to make a picture that exhibits the artifacts of 2022 Stable Diffusion-era images: the garbled text, the extra fingers, the uncanny faces.
None of that was ever chosen; it was what the models defaulted to.
Once the defaults moved past it, keeping the artifacts stopped being an accident and became a [decision](https://x.com/UnrealEngine/status/2113347611080151105).

This has happened before, in every medium that outgrew its defaults and the artifacts that come with them.
Pixel art spent the 80s as a hardware limitation nobody chose; once hardware moved past it, it became a genre people choose on purpose.
Auto-Tune was a hidden correction tool until Cher made it audible on Believe (1998); T-Pain turned it into a signature sound, and now it's a texture some singers reach for deliberately.
Film grain, tape hiss, VHS filters: defaults once, applied on purpose now.

![2022-era text-to-image outputs: garbled text and malformed shapes were the default]({{site.baseurl}}/assets/img/latent-diff-images.png)
*Source: [High-Resolution Image Synthesis with Latent Diffusion Models, Rombach et al. (2022)](https://arxiv.org/abs/2112.10752)*

Slop's artifacts don't disappear when the tools improve; they wait to be chosen. Give it a decade and a deliberately 2022-flavored image will read like 8-bit does today: a period style, with the slop bleached out of it.

---

## Decisions

"Is it AI or not" is the wrong question. It treats a tool as if it were a verdict.

The useful question is the one Chiang points at, just applied more carefully: **who made the decisions?** If the answer is "the model, by default", the result is slop, whether it came from a diffusion model, a preset pack, or a template. If the answer is "a person, deliberately, at whatever level of abstraction they chose to work at", then it's worth judging the work on its merits, like anything else.

Slop is what defaults look like. The fix has never been to ban the tool. It's to make more of the choices yourself.


> Give Shakespeare infinite monkeys  
> And see them, under his baton,  
> Write Hamlet

