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

Lectures are EDUCATIONAL content for STUDENTS across ALL DISCIPLINES:
- STEM, Sciences, Social Sciences, Humanities, Business, Medicine, Law, etc.
- Professor teaches domain-specific knowledge
- Students need comprehensive notes for LEARNING and EXAM PREPARATION

Break this into 6 ATOMIC tasks (ALL are EXECUTE nodes, NO further planning):

1. Transcribe lecture audio to text (SEARCH, EXECUTE)
   → Capture ALL verbal content with timestamps

2. Extract core concepts, definitions, and theoretical frameworks (THINK, EXECUTE, depends_on: [0])
   → UNIVERSAL: Works for ANY course domain
   → LLM identifies key terminology and foundational ideas

3. Extract structured/technical content (THINK, EXECUTE, depends_on: [0])
   → DOMAIN-ADAPTIVE: LLM identifies what's "structured" for THIS field
   → STEM: formulas, algorithms, code
   → Sciences: pathways, reactions, taxonomies
   → Humanities: analytical frameworks, rhetorical structures
   → Business: strategic models, financial frameworks
   → Return empty if pure discussion (valid!)

4. Extract examples, applications, and demonstrations (THINK, EXECUTE, depends_on: [0])
   → UNIVERSAL: All courses use examples
   → STEM: solved problems, code demos
   → Humanities: quotations, historical events, textual analysis
   → Sciences: case studies, experimental results
   → Business: company cases, market scenarios

5. Detect exam preparation signals and professor emphasis (THINK, EXECUTE, depends_on: [0])
   → UNIVERSAL: "This will be on the test", repeated topics, explicit importance cues

6. Generate comprehensive study guide (WRITE, EXECUTE, depends_on: [1,2,3,4])
   → Synthesize ALL extracted content into organized study material

**CRITICAL - LLM AUTONOMY:**
- LLM **autonomously detects** course domain (STEM vs Humanities vs Science vs Business)
- LLM **adapts extraction** to domain (formulas for physics, quotations for literature)
- LLM **determines importance** based on professor cues and content structure
- NO hardcoded quantities - extract EVERYTHING relevant
- Empty results are VALID (not all lectures have code/formulas/etc)

**Each sub_task MUST have:**
- "task_type": "SEARCH" | "THINK" | "WRITE"
- "node_type": "EXECUTE"  ← MANDATORY for ALL tasks!
- "depends_on_indices": [...]

Return JSON plan with sub_tasks array following SubTask schema."""


LECTURE_CONCEPT_PROMPT = """Extract ALL KEY CONCEPTS from this educational lecture.

**UNIVERSAL EXTRACTION - Works for ALL Academic Disciplines**

A concept is ANY of:
- New term or idea introduced
- Theory, framework, or model explained
- Important principle, rule, or law
- Technical/specialized terminology with definition
- Foundational ideas essential for understanding

TRANSCRIPT:
{transcript}

**For EACH concept extract:**
1. **term**: The concept name (preserve exact terminology)
2. **definition**: How professor defines it (exact wording preferred)
3. **category**: Type of concept (theory|principle|term|framework|method|law|model|etc)
4. **explanation**: Detailed explanation if provided
5. **domain_context**: What field/subfield this belongs to
6. **examples**: Real-world examples given (if any)
7. **prerequisites**: Related concepts students should know
8. **importance**: exam_critical|important|supplementary (based on professor cues)

**DOMAIN ADAPTIVITY:**
- STEM: Technical precision, mathematical rigor
- Sciences: Systematic relationships, processes
- Humanities: Interpretive frameworks, critical concepts
- Social Sciences: Theoretical models, methodologies
- Business: Strategic concepts, frameworks

**CRITICAL RULES:**
- Extract BOTH simple AND complex concepts
- Preserve technical accuracy (professor's exact definitions)
- Include domain-specific terminology
- Note prerequisites ("you should already know X")
- Empty list if lecture is purely applied/practical (valid!)

Return JSON:
{{
  "concepts": [
    {{
      "term": "...",
      "definition": "...",
      "category": "theory|principle|term|framework|method|etc",
      "explanation": "...",
      "domain_context": "...",
      "examples": ["..."],
      "prerequisites": ["..."],
      "importance": "exam_critical|important|supplementary"
    }}
  ],
  "total_concepts": 0,
  "detected_domain": "STEM|Science|Social_Science|Humanities|Business|Mixed",
  "concept_density": "high|medium|low"
}}"""


UNIVERSAL_STRUCTURED_CONTENT_PROMPT = """Extract ALL structured, technical, or methodological content from this lecture.

**AUTONOMOUS DOMAIN ADAPTATION:**
The lecture domain is UNKNOWN. Identify what constitutes "structured knowledge" for THIS specific field.

TRANSCRIPT:
{transcript}

**Domain-Specific Examples (LLM: adapt to actual lecture content):**

**STEM (Math/Physics/CS/Engineering):**
- Mathematical formulas, equations, derivations
- Algorithms, pseudocode, code implementations
- Theorems, proofs, problem-solving techniques
- Computational complexity, data structures
- Physical laws, chemical equations

**Natural Sciences (Biology/Chemistry/Medicine):**
- Metabolic pathways (verbal descriptions)
- Reaction mechanisms, chemical processes
- Classification systems, taxonomies
- Quantitative relationships (Michaelis-Menten, etc)
- Anatomical systems, physiological processes

**Social Sciences (Psychology/Sociology/Economics):**
- Experimental designs, statistical methods
- Theoretical frameworks, conceptual models
- Research methodologies, analysis techniques
- Economic models, market structures

**Humanities (Literature/Philosophy/History/Linguistics):**
- Analytical frameworks, critical lenses
- Rhetorical structures, literary devices
- Logical argument patterns, fallacies
- Historical frameworks (cause-effect, periodization)
- Linguistic structures, grammatical rules

**Business/Management:**
- Strategic frameworks (Porter's Five Forces, SWOT)
- Financial models, valuation methods
- Decision matrices, analytical tools

**For EACH structured item extract:**
1. **content**: The actual formula/framework/method (preserve notation)
2. **domain_type**: What type of content (formula/algorithm/framework/model/etc)
3. **components**: Key parts explained (variables, steps, elements)
4. **purpose**: What this is used for
5. **context**: When/how it's applied
6. **example**: Example usage (if provided)

**CRITICAL:**
- If NO structured content exists → return empty list (valid for discussion-based lectures)
- Extract EVERYTHING that requires systematic understanding
- Preserve precision (notation, terminology, exact wording)
- Don't invent - only extract what professor actually presented

Return JSON:
{{
  "structured_content": [
    {{
      "content": "...",
      "domain_type": "formula|algorithm|framework|model|pathway|etc",
      "components": {{}},
      "purpose": "...",
      "context": "...",
      "example": "..." (optional)
    }}
  ],
  "detected_domain": "STEM|Science|Social_Science|Humanities|Business|Mixed",
  "notes": "Any observations about content type"
}}"""


UNIVERSAL_EXAMPLES_APPLICATIONS_PROMPT = """Extract ALL examples, demonstrations, applications, and illustrative content from this lecture.

**AUTONOMOUS DOMAIN ADAPTATION:**
Identify what constitutes "examples" for THIS specific course domain.

TRANSCRIPT:
{transcript}

**Domain-Specific Example Types (LLM: extract what's present):**

**STEM (Math/CS/Engineering):**
- Solved problems with step-by-step solutions
- Code examples, algorithms, implementations
- Worked calculations, derivations
- Design examples, circuit diagrams (verbal descriptions)
- Case studies of technical systems

**Natural Sciences (Biology/Chemistry/Physics):**
- Case studies (diseases, reactions, phenomena)
- Experimental examples with results
- Real-world applications (drugs, technologies)
- Species examples, organism behaviors
- Lab procedure walkthroughs (verbal)

**Social Sciences (Psychology/Sociology/Economics):**
- Research studies, experiments
- Real-world case studies
- Statistical analysis examples
- Survey results, data interpretations
- Historical social phenomena

**Humanities (Literature/Philosophy/History):**
- Textual examples, quotations with analysis
- Historical events as illustrations
- Literary passages demonstrating concepts
- Philosophical thought experiments
- Comparative examples across periods/cultures

**Business/Management:**
- Company case studies
- Market scenarios, business situations
- Financial analysis examples
- Strategic decision examples
- Real-world business applications

**For EACH example extract:**
1. **type**: What kind of example (solved_problem|case_study|code|experiment|quotation|etc)
2. **content**: The actual example (preserve details)
3. **purpose**: What concept/principle this illustrates
4. **domain**: Subject area
5. **analysis**: Professor's explanation/interpretation
6. **key_takeaways**: What students should learn from this example

**CRITICAL:**
- Extract EVERYTHING used as illustration/demonstration
- Include professor's analysis/interpretation
- If no examples → empty list (valid for pure theory lectures)
- Preserve quotations, data, specifics exactly

Return JSON:
{{
  "examples_applications": [
    {{
      "type": "case_study|code|solved_problem|experiment|quotation|etc",
      "content": "...",
      "purpose": "Illustrates concept X",
      "domain": "...",
      "analysis": "Professor's interpretation",
      "key_takeaways": ["...", "..."]
    }}
  ],
  "total_examples": 0,
  "example_density": "high|medium|low|none"
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


UNIVERSAL_STUDY_GUIDE_PROMPT = """Create a COMPREHENSIVE STUDY GUIDE from this lecture data.

**AUTONOMOUS DOMAIN ADAPTATION:**
Adapt structure and format based on detected course domain. LLM decides optimal organization.

LECTURE DATA (from extractors):
{extracted_data}

**UNIVERSAL STRUCTURE (adapt sections based on content):**

# [Lecture Topic] - Study Guide

## 📚 Core Concepts & Definitions
[ALL key concepts - universal for any course]

## 🔬 Structured Knowledge (if present)
[Domain-adaptive content:]
- STEM: Formulas, equations, algorithms, code
- Sciences: Pathways, mechanisms, taxonomies, processes
- Humanities: Analytical frameworks, literary devices, rhetorical structures
- Social Sciences: Theoretical models, research methods, frameworks
- Business: Strategic models, financial frameworks, decision tools
[Skip section if empty - valid for discussion courses]

## 💡 Examples & Applications
[Real-world cases, demonstrations, solved problems, quotations, etc]

## ⚠️ Exam Preparation Focus
[Professor emphasis cues, repeated topics, exam-critical material]

## 📖 Additional Resources (if mentioned)
[Readings, supplementary materials]

## 📝 Assignments & Deadlines (if mentioned)
[Homework, projects, due dates]

---

**ADAPTIVE FORMATTING:**
- **STEM/Sciences**: Include step-by-step solutions, notation explanations
- **Humanities**: Include quotations, textual analysis, interpretive frameworks
- **Social Sciences**: Include research methodologies, statistical concepts
- **Business**: Include case analyses, strategic frameworks

**QUALITY REQUIREMENTS:**
- Use markdown for clarity
- Include timestamps for complex topics (e.g., "Topic X explained at 23:45")
- Highlight exam-critical content with ⚠️
- Organize hierarchically (main topics → sub-topics)
- **Adapt tone to domain** (technical for STEM, analytical for humanities)

**LLM AUTONOMY:**
- Decide optimal organization based on content
- Skip irrelevant sections (no formulas in literature → skip that section)
- Prioritize what students need for THIS specific course
- Create scannable, exam-focused structure

Write the comprehensive study guide now:"""


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
