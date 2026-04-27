# AI Learning OS

An AI Learning Chief of Staff with a Knowledge Brain — built as my AI Engineering Buildcamp capstone.

## The Problem

Ambitious self-learners and early-career professionals save large amounts of useful content across YouTube, notes apps, bookmarks, and articles, but struggle to turn that information into structured progress. Resources are scattered, priorities are unclear, and there is no system connecting saved content to a clear learning roadmap.

## What It Does

The AI Learning OS ingests and organizes saved content (YouTube links, notes, articles) into a searchable personal knowledge base, generates a goal-based learning roadmap, ranks the most relevant resources, and returns a weekly action plan with recommended next steps.

A typical interaction: the user provides their learning goal (e.g., "become job-ready in AI engineering in 4 months"), available study time, and a set of saved resources. They can then ask "What should I focus on this week?" and the system returns a personalized study plan, prioritized resources drawn from their own saved content, and progress tracking updates.

## Setup

1. Install uv if you don't have it yet: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository (or download the zip and extract it).

3. Create a `.env` file from the template and add your API key:

       cp .env.example .env

4. Install dependencies:

       uv sync

5. Start Jupyter:

       uv run jupyter notebook

## Notebooks

- `notebooks/01-setup.ipynb` - smoke test that confirms your environment works
- `notebooks/02-rag.ipynb` - a minimal RAG baseline you can adapt to your own data

## Data

Put your project data in the `data/` folder. See `notebooks/02-rag.ipynb` for how to load it.
