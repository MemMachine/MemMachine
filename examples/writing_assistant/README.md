# Writing Assistant

A writing assistant that analyzes and learns your writing style, then generates new content in your unique voice.

## Features

- **Writing Style Analysis**: Analyzes your writing samples to extract detailed style characteristics
- **Content Type Support**: Separate style profiles for different content types (email, blog, LinkedIn, etc.)
- **Style Matching**: Generates new content that matches your established writing patterns
- **Magic Keyword Support**: Use `/submit` command to easily submit writing samples

## How It Works

### 1. Submit Writing Samples

Use the `/submit` command followed by the content type and your writing sample:

```
/submit email Dear John, I hope this finds you well. I wanted to follow up on our meeting yesterday.
```

```
/submit blog The future of technology is bright and full of possibilities. As we look ahead...
```

```
/submit linkedin Excited to share our latest product launch! This represents months of hard work...
```

```
/submit general This is a sample of my general writing style for various contexts.
```

**Supported Content Types:**
- email
- blog
- linkedin
- twitter
- facebook
- instagram
- report
- proposal
- memo
- letter
- essay
- article
- creative
- technical
- formal
- casual
- general (default if no type specified)

### 2. Writing Style Analysis

The assistant analyzes your writing samples across 24 different style characteristics:

- **tone** - Overall emotional quality and attitude
- **register** - Formality level and appropriateness
- **voice** - Distinctive personality and perspective
- **sentence_structure** - Sentence patterns and complexity
- **pacing** - Rhythm and flow of content
- **word_choice** - Vocabulary sophistication and selection
- **parts_of_speech_tendency** - Preference for certain parts of speech
- **tense_usage** - Primary tense and voice preferences
- **grammar_quirks** - Unique grammatical patterns
- **clarity** - Directness and straightforwardness
- **logic_and_flow** - Logical progression of ideas
- **cohesion_devices** - Transitional phrases and connections
- **paragraphing_style** - Paragraph structure and organization
- **rhetorical_devices** - Use of metaphors, repetition, etc.
- **use_of_examples** - Frequency and type of examples
- **directness** - Straightforward vs. indirect communication
- **personality** - Humorous vs. serious approach
- **humor_style** - Type and frequency of humor
- **emotional_intensity** - Level of emotional expression
- **self_reference** - Use of first person and personal anecdotes
- **signature_phrases** - Frequently used expressions
- **patterned_openings_or_closings** - Consistent greeting/farewell styles
- **motifs_or_themes** - Recurring themes or concepts
- **use_of_headings_subheadings** - Structural organization preferences

### 3. Content Generation

Once you've submitted writing samples, the assistant can generate new content in your style:

```
Write an email to my team about the project deadline
```

```
Create a LinkedIn post about our company's achievements
```

```
Help me write a blog post about remote work trends
```

## API Endpoints

### POST `/memory`
Store user messages and detect writing style submissions.

**Request:**
```json
{
  "user_id": "user123",
  "query": "/submit email Dear John, I hope this finds you well."
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Writing sample for email content type submitted successfully",
  "data": {...}
}
```

### GET `/memory`
Retrieve memory data and format for writing assistant.

**Parameters:**
- `query`: The user's query
- `user_id`: User identifier
- `timestamp`: Optional timestamp

### POST `/memory/store-and-search`
Store user data and immediately search for relevant context.

### POST `/analyze-writing-style`
Analyze a writing sample and extract style characteristics.

### GET `/writing-styles/{user_id}`
Get all writing styles for a specific user.

**Response:**
```json
{
  "status": "success",
  "user_id": "user123",
  "writing_styles": {
    "email": [
      {
        "feature": "tone",
        "value": "professional and warm",
        "tag": "writing_style_email"
      }
    ]
  },
  "available_content_types": ["email", "blog", "linkedin"]
}
```

## Configuration

Set these environment variables:

- `MEMORY_BACKEND_URL`: URL of the MemMachine backend (default: http://localhost:8080)
- `WRITING_ASSISTANT_PORT`: Port for the writing assistant server (default: 8000)

## Running the Server

```bash
cd examples/writing_assistant
python writing_assistant_server.py
```

## Testing

Run the test suite to verify functionality:

```bash
python test_writing_assistant.py
```

## Usage Examples

### Example 1: Submit an Email Sample
```
/submit email Hi Sarah, I wanted to follow up on our discussion about the quarterly review. I think we should schedule a meeting to go over the details and make sure we're aligned on the timeline.
```

### Example 2: Submit a Blog Sample
```
/submit blog The digital transformation journey is not just about technology—it's about people, processes, and culture. Organizations that succeed in this space understand that change management is just as important as the technical implementation.
```

### Example 3: Generate Content in Your Style
```
Write an email to my manager about requesting time off for vacation
```

The assistant will analyze your existing email writing style and generate content that matches your tone, formality level, sentence structure, and other characteristics.

## ProfileMemory Integration

The writing assistant integrates with MemMachine's ProfileMemory system to:

1. **Store Style Characteristics**: Each writing sample is analyzed and stored as profile entries with the tag format `writing_style_{content_type}`
2. **Retrieve Style Data**: When generating content, the assistant retrieves relevant style characteristics for the requested content type
3. **Update Profiles**: New writing samples update and refine existing style profiles
4. **Maintain Separation**: Different content types maintain separate style profiles to preserve context-appropriate writing patterns

This ensures that your writing style is preserved and consistently applied across all generated content while maintaining the flexibility to adapt to different content types and contexts.
