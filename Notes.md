Improvements
1. Update LLM instructions (orchestrator) for each conversation file to note that the counterparty is always the user.
2. Some answers with llama 3.1:8b are too literal ("What messages contain anything about Anthropic", shows the message titles, not the actual messages. Need to specifically indicate to the model to show the message contents.)

What works well:
1. source lists
2. seems to handle Greeklish fairly well (still need more messages to test).