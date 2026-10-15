# Intern Challenge

Source: Humanoid, "Internship, Robot Learning Research" job posting (https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221). Text below is copied from the posting; sections not relevant to the challenge are left out.

## How to apply

Complete the challenge below and submit your solution as a public GitHub repository. You will be able to include your GitHub repository URL when you fill out the application form, alongside your name and CV. You have two weeks to complete the challenge and submit your solution. The deadline for submission is Friday, 16 October 2026, 23:59 BST.

We're not looking for standard solutions, we're looking for how you think. The strongest submissions are creative, original, and push beyond the obvious.

## Intern Challenge

The goal of the challenge is to use real data collected by an applicant to drive a robotic manipulator in a simple simulation environment (e.g. Libero). The applicant is welcome to use a simple phone to record a small manipulation dataset and use it creatively showcasing their knowledge with VLA and/or World Models.

Here is an example of how it might look like:

Left: a snapshot from an egocentric hand-manipulation video sample; Right: simulation environment where we use that data to drive a Panda arm in Libero simulator with a trained policy based on SmolVLA.

Some suggestions on how you can develop your project:

- use your recorded egocentric data to post-train a policy on a simple task
- explore creative retargeting strategies, e.g. adapt your data to challenging embodiments
- bootstrap a policy and use any form of RL to get better performance
- use world modelling to showcase video/state prediction, less focusing on policy performance
- optimise a standard policy to run considerably faster than a baseline

However, we don't want to limit your imagination: in the age of AI agents, a standard task can now be easily achieved. You are welcome to use any resources at your disposal as long as the main constraint is achieved: you use the data that you personally collected. At the same time, we tried to design the challenge so that it could be hand-coded as well. We also checked that many ideas do not require access to large compute and could be done on Google Colab GPU notebooks.

## What to submit

Complete the challenge above and submit your solution as a public GitHub repository by Friday, 16 October 2026, 23:59 BST. Include a README/Presentation with instructions to run your system, example outputs, and a note on your design choices, what worked and what didn't.

## What we care about

- Creativity in approach, while satisfying the constraint: the data you collected must play a role in the approach
- Performance of your policy in simulation and/or quality of WM predictions
- Implementation simplicity and clear presentation of results without AI slop

Make something you're proud of!

## Related context from the posting: World Models

What the team says interns might work on in this area:

- Action-conditioned video prediction and dynamics models that stay physically consistent over long horizons
- Use world models as learned simulators: score candidate policies offline and generate synthetic rollouts for training
- Build fidelity metrics that quantify where the world model can be trusted
