EXTRACTION_PROMPT = """\
You are an expert cognitive memory extraction system.
Analyze the following session conversation transcript and extract two types of memory:

1. EPISODIC MEMORY: Key events, user goals, major interactions, and a concise summary timeline of what transpired in this session.
2. SEMANTIC MEMORY: Concrete facts, domain knowledge, user preferences, terms, or key insights established during the conversation.

Format your response as valid JSON with exact keys:
{{
  "episodic_summary": "...",
  "key_events": ["..."],
  "semantic_facts": ["..."],
  "user_preferences": ["..."]
}}

Transcript:
{formatted_transcript}
"""