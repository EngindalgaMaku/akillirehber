# Design Document

## Overview

This design enhances the course settings system with improved UI/UX and adds course-specific system prompt functionality. The solution involves database schema updates, API enhancements, and a redesigned frontend interface with better visual hierarchy and user experience.

## Architecture

The enhancement follows the existing three-tier architecture:

1. **Database Layer**: Add system_prompt field to CourseSettings model
2. **API Layer**: Update schemas and endpoints to handle system prompts
3. **Frontend Layer**: Redesign settings UI with improved visual design and system prompt editor

## Components and Interfaces

### Database Schema Changes

**CourseSettings Model Enhancement:**
```python
class CourseSettings(Base):
    # ... existing fields ...
    system_prompt = Column(Text, nullable=True)  # New field for course-specific prompts
```

**Migration Requirements:**
- Add system_prompt column to course_settings table
- Set default system prompt for existing courses
- Ensure proper indexing and constraints

### API Schema Updates

**CourseSettingsResponse Schema:**
```python
class CourseSettingsResponse(BaseModel):
    # ... existing fields ...
    system_prompt: Optional[str] = None
```

**CourseSettingsUpdate Schema:**
```python
class CourseSettingsUpdate(BaseModel):
    # ... existing fields ...
    system_prompt: Optional[str] = Field(None, max_length=2000)
```

### Frontend Component Structure

**Enhanced Settings UI Layout:**
```
┌─────────────────────────────────────┐
│ Course Settings Header              │
├─────────────────────────────────────┤
│ System Prompt Section               │
│ ┌─────────────────────────────────┐ │
│ │ Large textarea with counter     │ │
│ │ Character limit: 2000           │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ Chunking Settings Section           │
│ ┌─────────────┬─────────────────┐   │
│ │ Strategy    │ Chunk Size      │   │
│ │ Overlap     │ Embedding Model │   │
│ └─────────────┴─────────────────┘   │
├─────────────────────────────────────┤
│ Search Settings Section             │
│ ┌─────────────┬─────────────────┐   │
│ │ Hybrid Alpha│ Top-K Results   │   │
│ └─────────────┴─────────────────┘   │
├─────────────────────────────────────┤
│ LLM Settings Section                │
│ ┌─────────────┬─────────────────┐   │
│ │ Provider    │ Model           │   │
│ │ Temperature │ Max Tokens      │   │
│ └─────────────┴─────────────────┘   │
├─────────────────────────────────────┤
│ Save Button                         │
└─────────────────────────────────────┘
```

## Data Models

### Default System Prompt

**Default Prompt Content:**
```
Sen yardımcı bir eğitim asistanısın. Öğrencilere ve öğretmenlere ders materyalleri hakkında açık, doğru ve eğitici yanıtlar veriyorsun. 

Görevlerin:
- Ders içeriğini anlaşılır şekilde açıklamak
- Öğrenci sorularını sabırla yanıtlamak  
- Kaynak materyallere dayalı bilgi vermek
- Öğrenmeyi teşvik edici bir ton kullanmak

Her zaman:
- Nazik ve profesyonel ol
- Eğitici ve yapıcı yanıtlar ver
- Kaynaklarını belirt
- Anlaşılır dil kullan
```

### System Prompt Validation Rules

1. **Length Validation**: Maximum 2000 characters
2. **Content Validation**: No malicious instructions
3. **Format Validation**: Plain text only
4. **Fallback Behavior**: Use default if empty or invalid

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: System Prompt Persistence
*For any* course settings update containing a system prompt, saving the settings should result in the system prompt being retrievable in subsequent requests.
**Validates: Requirements 2.3**

### Property 2: Default Prompt Assignment
*For any* newly created course, the course should automatically have the default system prompt assigned.
**Validates: Requirements 3.2**

### Property 3: Character Limit Enforcement
*For any* system prompt input exceeding 2000 characters, the system should reject the input and display an appropriate error message.
**Validates: Requirements 5.1**

### Property 4: Chat Integration
*For any* chat request to a course, the system prompt should be included in the LLM context if a custom prompt exists, otherwise the default prompt should be used.
**Validates: Requirements 4.1, 4.3**

### Property 5: Settings UI Validation
*For any* system prompt field interaction, the character counter should accurately reflect the current input length and warn when approaching the limit.
**Validates: Requirements 5.4**

## Error Handling

### System Prompt Validation Errors
- **Character Limit Exceeded**: Display warning at 1900 characters, error at 2000+
- **Invalid Content**: Sanitize input and provide feedback
- **Save Failures**: Show specific error messages and retry options

### UI Error States
- **Loading States**: Show spinners during save operations
- **Network Errors**: Display retry buttons and error messages
- **Validation Errors**: Highlight problematic fields with clear messages

### Fallback Mechanisms
- **Missing System Prompt**: Use default prompt automatically
- **Corrupted Prompt**: Reset to default with user notification
- **API Failures**: Cache changes locally and retry

## Testing Strategy

### Unit Tests
- System prompt validation logic
- Character counting functionality
- Default prompt assignment
- API endpoint parameter validation

### Property-Based Tests
- System prompt persistence across save/load cycles
- Character limit enforcement with random inputs
- Default prompt assignment for course creation
- Chat integration with various prompt configurations

### Integration Tests
- End-to-end settings save and retrieval
- Chat functionality with custom system prompts
- UI component interactions and state management
- Database migration and data integrity

### UI/UX Tests
- Visual regression tests for improved layout
- Accessibility compliance testing
- Responsive design validation
- User interaction flow testing

**Testing Configuration:**
- Minimum 100 iterations per property test
- Each property test references its design document property
- Tag format: **Feature: course-settings-enhancement, Property {number}: {property_text}**