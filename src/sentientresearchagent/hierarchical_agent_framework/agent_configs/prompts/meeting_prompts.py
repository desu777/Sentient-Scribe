"""
Meeting-specific prompts for MeetingGenius agents.

Type-aware prompts optimized for different meeting contexts:
- Agile Standups: Tactical, blockers, immediate actions
- Lectures: Educational, concepts, exam prep
- Sales Calls: Pain points, objections, next steps
- Retrospectives: Process improvement, team feedback

Each prompt is optimized for the specific meeting type's objectives.
"""

from datetime import datetime, timezone

CURRENT_DATE = datetime.now(timezone.utc).strftime('%Y-%m-%d')


# ============================================================================
# STANDUP MEETING PROMPTS
# ============================================================================

STANDUP_MEETING_PLANNER_PROMPT = """You are planning the processing of an Agile Daily Standup meeting.

Standups are SHORT (15-30 min), TACTICAL meetings where:
- Team members share: Yesterday's work, Today's plans, Blockers
- Focus is on IMMEDIATE actions and removing obstacles
- NOT strategic planning (that's for sprint planning)

Break this into 6 ATOMIC tasks (ALL are EXECUTE nodes, NO further planning):

1. Transcribe audio to text (SEARCH, EXECUTE)
2. Extract blockers (THINK, EXECUTE, depends_on: [0])
3. Extract action items (THINK, EXECUTE, depends_on: [0])
4. Extract accomplishments (THINK, EXECUTE, depends_on: [0])
5. Extract today's plans (THINK, EXECUTE, depends_on: [0])
6. Generate summary (WRITE, EXECUTE, depends_on: [1,2,3,4])

CRITICAL: Each sub_task MUST have:
- "task_type": "SEARCH" | "THINK" | "WRITE"
- "node_type": "EXECUTE"  ← MANDATORY for ALL tasks!
- "depends_on_indices": [...]

Return JSON plan with sub_tasks array following SubTask schema."""


STANDUP_ACTION_PROMPT = """You are an expert at extracting action items from Agile standup meetings.

STANDUP CONTEXT:
- Fast-paced, tactical meetings
- Actions are SHORT-TERM (today, tomorrow, this sprint)
- Focus on UNBLOCKING progress and coordinating work

TRANSCRIPT:
{transcript}

ATTENDEES (if known): {attendees}

TASK:
Extract ALL action items mentioned. For each:

1. **task**: Specific actionable task (not vague)
2. **owner**: Person responsible (exact name from transcript)
3. **deadline**: When it's due (parse: "today", "tomorrow", "Friday", "end of sprint")
4. **priority**:
   - HIGH: Blocks deployment or other team members
   - MEDIUM: Important but not blocking
   - LOW: Nice-to-have or long-term
5. **context**: WHY it matters (1 sentence)

EXTRACTION RULES:
- Include even implicit actions (e.g., "I'll help Sarah debug" = action item)
- If no deadline mentioned → use "Not specified"
- Focus on CODE-RELATED tasks (PRs, deployments, bug fixes)
- Ignore vague commitments ("I'll think about it")

Return ONLY valid JSON:
{{
  "items": [
    {{
      "task": "Review PR #234 for login feature",
      "owner": "Mike",
      "deadline": "Before merge (today)",
      "priority": "HIGH",
      "context": "Blocks John's deployment to staging"
    }}
  ]
}}"""


STANDUP_BLOCKER_PROMPT = """Extract BLOCKERS from this standup transcript.

BLOCKER = Anything preventing someone from making progress.

Common blockers:
- Waiting for code review
- Technical issue (bug, env problem, failing tests)
- Waiting for another team/person
- Missing information/access/credentials
- External dependency (API down, service issue)

TRANSCRIPT:
{transcript}

For EACH blocker extract:
1. **who**: Person who is blocked
2. **blocker**: What's blocking them (specific)
3. **impact**: What can't be done because of this
4. **priority**:
   - CRITICAL: Blocks deployment or sprint goal
   - HIGH: Blocks current task
   - MEDIUM: Slows progress
5. **who_can_help**: If mentioned, who can unblock this

Return JSON:
{{
  "blockers": [
    {{
      "who": "Sarah",
      "blocker": "API authentication keeps failing in staging",
      "impact": "Can't test user flow, blocks Friday release",
      "priority": "CRITICAL",
      "who_can_help": "John (API expert)"
    }}
  ]
}}"""


# ============================================================================
# LECTURE PROMPTS
# ============================================================================

LECTURE_PLANNER_PROMPT = """You are planning the processing of an educational lecture recording.

Lectures are EDUCATIONAL content where:
- Professor teaches concepts, theories, methods
- Students need to LEARN and PREPARE FOR EXAMS
- Focus is on comprehensive understanding and exam preparation

Break this into 6 ATOMIC tasks (ALL are EXECUTE nodes, NO further planning):

1. Transcribe lecture audio to text (SEARCH, EXECUTE)
2. Extract key concepts and definitions (THINK, EXECUTE, depends_on: [0])
3. Extract formulas, equations, algorithms (THINK, EXECUTE, depends_on: [0])
4. Extract code examples and technical demonstrations (THINK, EXECUTE, depends_on: [0])
5. Detect exam preparation hints and important topics (THINK, EXECUTE, depends_on: [0])
6. Generate comprehensive study guide (WRITE, EXECUTE, depends_on: [1,2,3,4])

CRITICAL INSTRUCTIONS:

**LLM AUTONOMY:**
- LLM decides WHAT to extract (not hardcoded categories)
- LLM determines IMPORTANCE (what's exam-critical vs supplementary)
- LLM identifies STRUCTURE (how to organize concepts)
- Extract EVERYTHING relevant, not fixed quantities

**EXTRACTION FLEXIBILITY:**
- If no formulas → empty list (not error)
- If no code → empty list (valid for humanities)
- Adapt to course type automatically (STEM vs humanities vs business)

**Each sub_task MUST have:**
- "task_type": "SEARCH" | "THINK" | "WRITE"
- "node_type": "EXECUTE"  ← MANDATORY for ALL tasks!
- "depends_on_indices": [...]

Return JSON plan with sub_tasks array following SubTask schema."""


LECTURE_CONCEPT_PROMPT = """Extract KEY CONCEPTS from this educational lecture.

A concept is:
- New term or idea introduced
- Theory or framework explained
- Important principle or rule
- Technical term with definition

TRANSCRIPT:
{transcript}

COURSE: {course_topic}

For EACH concept extract:
1. **term**: The concept name
2. **definition**: How it's defined (in professor's words)
3. **category**: Type (theory/algorithm/principle/term/framework)
4. **explanation**: Longer explanation if provided
5. **example**: Real-world example given (if any)
6. **importance**: Is this flagged as exam-critical?

RULES:
- Include BOTH simple and complex concepts
- Preserve technical accuracy (don't simplify definitions)
- Note prerequisites (concepts mentioned as "you should already know...")

Return JSON:
{{
  "concepts": [
    {{
      "term": "Backpropagation",
      "definition": "Algorithm for calculating gradients in neural networks",
      "category": "algorithm",
      "explanation": "Uses chain rule to compute partial derivatives...",
      "example": "Training a network to recognize handwritten digits",
      "importance": "exam_critical"
    }}
  ],
  "prerequisites_mentioned": ["Calculus", "Linear Algebra"]
}}"""


LECTURE_FORMULA_PROMPT = """Extract all MATHEMATICAL FORMULAS and EQUATIONS from lecture.

TRANSCRIPT:
{transcript}

For EACH formula:
1. **formula**: The equation (try to format as LaTeX if possible)
2. **variables**: What each variable represents
3. **purpose**: What this formula calculates
4. **example**: Example calculation (if given)
5. **when_to_use**: Application context

Example extraction:
{{
  "formulas": [
    {{
      "formula": "dL/dw = dL/da * da/dz * dz/dw",
      "variables": {{
        "L": "Loss function",
        "w": "Weights",
        "a": "Activation",
        "z": "Pre-activation"
      }},
      "purpose": "Calculate gradient for backpropagation",
      "example": "For 2-layer network: ...",
      "when_to_use": "Training neural networks"
    }}
  ]
}}"""


LECTURE_CODE_PROMPT = """Extract CODE EXAMPLES from this lecture transcript.

TRANSCRIPT:
{transcript}

For EACH code example:
1. **language**: Programming language
2. **code**: Full code snippet (preserve formatting)
3. **purpose**: What this code demonstrates
4. **key_points**: Important aspects highlighted
5. **common_errors**: Mistakes to avoid (if mentioned)

Return JSON:
{{
  "code_examples": [
    {{
      "language": "Python",
      "code": "def backprop(x, y, w):\\n    z = np.dot(x, w)\\n    a = sigmoid(z)\\n    return (a - y)**2",
      "purpose": "Demonstrates backpropagation calculation",
      "key_points": ["Uses numpy for efficiency", "Sigmoid activation"],
      "common_errors": ["Forgetting to transpose matrices"]
    }}
  ]
}}"""


EXAM_HINT_DETECTOR_PROMPT = """Identify EXAM PREPARATION HINTS from lecture.

Look for phrases indicating exam importance:
- "This WILL BE on the test"
- "Make sure you understand this"
- "Common exam question"
- "You should memorize..."
- "Important for the midterm"
- "I always ask about this"

TRANSCRIPT:
{transcript}

For EACH exam hint:
1. **topic**: What concept/topic
2. **hint_phrase**: Exact phrase used
3. **importance_level**: "must_know" | "important" | "helpful"
4. **exam_type**: "midterm" | "final" | "quiz" | "general"

Return JSON:
{{
  "exam_hints": [
    {{
      "topic": "Backpropagation algorithm",
      "hint_phrase": "This WILL BE ON THE MIDTERM EXAM",
      "importance_level": "must_know",
      "exam_type": "midterm"
    }}
  ]
}}"""


HOMEWORK_ASSIGNMENT_DETECTOR_PROMPT = """Extract HOMEWORK ASSIGNMENTS from lecture.

Look for:
- Specific tasks to complete
- Problem sets to solve
- Readings to do
- Projects to work on

TRANSCRIPT:
{transcript}

For EACH assignment:
1. **assignment**: What to do
2. **deadline**: When it's due
3. **deliverable**: What to submit
4. **resources**: Required materials/tools

Return JSON:
{{
  "assignments": [
    {{
      "assignment": "Implement backpropagation for 3-layer neural network",
      "deadline": "Next Friday",
      "deliverable": "Python code + written explanation",
      "resources": ["NumPy library", "Lecture notes section 4.2"]
    }}
  ]
}}"""


# ============================================================================
# GENERIC MEETING PROMPTS (used by multiple types)
# ============================================================================

DECISION_EXTRACTION_PROMPT = """Extract KEY DECISIONS made during this meeting.

A decision is:
- Choice between alternatives ("We'll use PostgreSQL, not MySQL")
- Commitment to action ("We're postponing mobile app to Q2")
- Policy/process change ("New code review process approved")
- Budget/resource allocation ("Approved $5K for cloud infra")

NOT a decision:
- Ongoing discussions without conclusion
- Individual tasks (those are action items)
- Informational updates

TRANSCRIPT:
{transcript}

For EACH decision:
1. **decision**: What was decided
2. **rationale**: Why (if mentioned)
3. **who_decided**: Who made or approved decision
4. **impact**: What this affects

Return JSON:
{{
  "decisions": [
    {{
      "decision": "Use PostgreSQL for new microservice database",
      "rationale": "Better JSON support and team familiarity",
      "who_decided": "Team consensus, John advocated",
      "impact": "Affects architecture and deployment timeline"
    }}
  ]
}}"""


SPEAKER_ANALYSIS_PROMPT = """Analyze SPEAKER CONTRIBUTIONS in this meeting.

TRANSCRIPT:
{transcript}

KNOWN ATTENDEES: {attendees}

For EACH speaker identified:
1. **name**: Speaker name (or Speaker 1, Speaker 2 if unknown)
2. **speaking_time_estimate**: Rough estimate (short/medium/long or minutes if calculable)
3. **key_points**: Main points they made (3-5 bullet points)
4. **role_in_meeting**: Leader, participant, quiet, dominating
5. **action_items_received**: Tasks assigned to them

Return JSON:
{{
  "speakers": [
    {{
      "name": "John",
      "speaking_time_estimate": "5 minutes",
      "key_points": [
        "Completed login feature",
        "Ready to deploy to staging",
        "Proposed PostgreSQL for database"
      ],
      "role_in_meeting": "Active contributor",
      "action_items_received": 2
    }}
  ]
}}"""


# ============================================================================
# SUMMARY WRITERS (type-specific)
# ============================================================================

STANDUP_SUMMARY_PROMPT = """Write a TACTICAL summary of this standup meeting.

STANDUP DATA (from extractors):
{extracted_data}

Write a concise 2-3 paragraph summary focusing on:
1. KEY BLOCKERS (most critical first)
2. Important accomplishments from yesterday
3. High-priority action items for today/tomorrow
4. Team velocity signals (are we on track?)

FORMAT:
- Bullet points for blockers and actions
- Short paragraphs (2-3 sentences max)
- Tactical language (not strategic)
- Highlight URGENT items

Example:
"**Critical Blockers:** Sarah is blocked on API authentication (blocks Friday release).
Mike is waiting for design approval.

**Progress:** John completed login feature and is ready to deploy. Lisa finished UI mockups.

**Today's Focus:** Fix API bug (Sarah + John pair), review PR #234 (Mike), update docs (all).

**Action Items:** 8 items assigned, 3 high-priority, all due by EOD Friday."

Write standup summary now:"""


STUDY_GUIDE_PROMPT = """Create a COMPREHENSIVE STUDY GUIDE from this lecture data.

LECTURE DATA (from extractors):
{extracted_data}

Structure the study guide with these sections:

# [Lecture Topic] - Study Guide

## 📚 Key Concepts
[List all concepts with definitions]

## 📐 Formulas & Equations
[All formulas with variable explanations]

## 💻 Code Examples
[Code snippets with explanations]

## ⚠️ Exam Preparation
[Flagged topics that "will be on the test"]

## ✅ Practice Problems
[Any example problems solved in lecture]

## 📖 Recommended Reading
[Additional resources mentioned]

## 📝 Homework
[Assignments with deadlines]

---

FORMAT REQUIREMENTS:
- Use markdown formatting
- Include timestamps for complex topics (e.g., "Backpropagation explained at 1:23:45")
- Highlight exam-critical content with ⚠️ emoji
- Organize by topic/chapter if lecture covers multiple areas

Write the study guide now:"""


# ============================================================================
# AGGREGATOR PROMPTS
# ============================================================================

STANDUP_DATA_AGGREGATOR_PROMPT = """Combine these standup extraction results into unified structured data.

You received parallel extractions:
- Blockers: {blocker_results}
- Action items: {action_results}
- Yesterday's work: {accomplishments}
- Today's plans: {plans}

Create a UNIFIED standup data structure:
{{
  "blockers": [list of all blockers],
  "action_items": [list of all actions],
  "accomplishments": [list by person],
  "today_plans": [list by person],
  "summary_stats": {{
    "total_blockers": X,
    "critical_blockers": Y,
    "total_action_items": Z,
    "high_priority_actions": W
  }}
}}

Ensure NO duplication (same item mentioned in multiple extractions).
Cross-reference: If action item addresses a blocker, note the connection."""


LECTURE_DATA_AGGREGATOR_PROMPT = """Combine these lecture extraction results into unified study material.

You received:
- Concepts: {concept_results}
- Formulas: {formula_results}
- Code examples: {code_results}
- Exam hints: {exam_hints}
- Assignments: {assignments}

Create UNIFIED lecture data:
{{
  "concepts": [all concepts, remove duplicates],
  "formulas": [all formulas with context],
  "code_examples": [all code with explanations],
  "exam_critical_topics": [topics flagged for exam],
  "assignments": [homework with deadlines],
  "topic_organization": [group related concepts],
  "estimated_study_time": "X hours based on complexity"
}}

Connect related items (e.g., formula used in code example)."""


# ============================================================================
# DELIVERABLE WRITERS
# ============================================================================

SLACK_MESSAGE_FORMATTER_PROMPT = """Format standup summary for Slack channel #standup.

DATA:
{standup_data}

Create a Slack-optimized message:
- Use Slack markdown (bold with *, code with `)
- Keep it concise (readable on mobile)
- Use emojis sparingly (🚨 for critical blockers, ✅ for done)
- Tag people with action items (@john)
- Structure with clear sections

Example format:
```
📋 *Daily Standup Summary* - {date}

🚨 *Critical Blockers*
• Sarah - API auth failing (blocks deployment) - needs @john help

✅ *Yesterday's Wins*
• @john - Shipped login feature
• @lisa - Finished mockups

🎯 *Today's Focus*
• Fix API bug (Sarah + John)
• Review PR #234 (Mike)
• Deploy to staging (John)

📌 *Action Items* (8 total, 3 high-priority)
[See thread for details]
```

Write Slack message:"""


EMAIL_FOLLOW_UP_PROMPT = """Draft a professional follow-up email after the meeting.

MEETING DATA:
{meeting_data}

RECIPIENTS: {attendees}

Create email with:

Subject: Action Items from [Meeting Type] - [Date]

Body:
- Brief meeting summary (2-3 sentences)
- Key decisions made (if any)
- Action items per person (clear, specific)
- Next meeting/deadline reminders
- Professional but friendly tone

Make it SCANNABLE:
- Use bullet points
- Bold important items
- Clear sections
- Short paragraphs

Write the email:"""


# ============================================================================
# INTERVIEW/PODCAST EXTRACTION PROMPTS (for testing with Ronaldo podcast)
# ============================================================================

INSIGHTS_EXTRACTION_PROMPT = """Extract KEY INSIGHTS from this interview/podcast transcript.

An insight is:
- Philosophy or principle shared
- Methodology or approach explained
- Personal experience or lesson learned
- Advice or recommendation given
- Surprising or counter-intuitive fact

TRANSCRIPT:
{transcript}

For EACH insight extract:
1. **category**: Type (philosophy/methodology/advice/experience/fact)
2. **insight**: The main insight (1-2 sentences)
3. **quote**: Exact quote from transcript (if applicable)
4. **speaker**: Who shared this insight
5. **context**: Additional context or explanation
6. **importance**: "high" | "medium" | "low"

Focus on ACTIONABLE insights (things audience can learn/apply).

Return JSON:
{{
  "insights": [
    {{
      "category": "philosophy",
      "insight": "Talent and hard work must combine for sustained excellence",
      "quote": "Talent without work is nothing, and work without talent is nothing, too",
      "speaker": "Cristiano Ronaldo",
      "context": "Discussing what made him successful for 20 years at the top",
      "importance": "high"
    }}
  ],
  "total_insights": 8
}}"""


# ============================================================================
# HELPER PROMPTS
# ============================================================================

TOPIC_EXTRACTION_PROMPT = """Identify main TOPICS discussed in this meeting.

TRANSCRIPT:
{transcript}

For each topic:
1. **topic**: Topic name
2. **time_spent**: Estimated minutes spent on this
3. **key_points**: 2-3 main points discussed
4. **resolution**: Resolved | Ongoing | Tabled

Return JSON:
{{
  "topics": [
    {{
      "topic": "Sprint Planning",
      "time_spent": "18 minutes",
      "key_points": ["Assigned 12 story points", "Velocity tracking", "Dependencies identified"],
      "resolution": "Resolved"
    }}
  ]
}}"""
