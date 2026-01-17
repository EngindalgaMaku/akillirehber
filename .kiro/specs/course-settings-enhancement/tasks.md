# Implementation Plan: Course Settings Enhancement

## Overview

This implementation plan enhances the course settings system with improved UI design and adds course-specific system prompt functionality. The approach follows a backend-first strategy to ensure data integrity before updating the frontend interface.

## Tasks

- [x] 1. Database Schema and Migration
  - Add system_prompt field to CourseSettings model
  - Create database migration with default system prompt
  - Update model relationships and constraints
  - _Requirements: 2.1, 3.2_

- [ ] 2. Backend API Updates
  - [x] 2.1 Update CourseSettings database model
    - Add system_prompt Text field to CourseSettings class
    - Set appropriate nullable and default constraints
    - _Requirements: 2.1, 2.3_

  - [x] 2.2 Update API schemas
    - Add system_prompt field to CourseSettingsResponse schema
    - Add system_prompt field to CourseSettingsUpdate schema with validation
    - Add character limit validation (max 2000 characters)
    - _Requirements: 2.3, 5.1_

  - [x] 2.3 Update course creation logic
    - Modify course creation to assign default system prompt
    - Ensure CourseSettings are created with default values
    - _Requirements: 3.2_

  - [ ] 2.4 Write property test for system prompt persistence
    - **Property 1: System Prompt Persistence**
    - **Validates: Requirements 2.3**

- [ ] 3. Default System Prompt Configuration
  - [ ] 3.1 Define default system prompt constant
    - Create Turkish educational assistant prompt
    - Include helpful, educational behavior instructions
    - _Requirements: 3.1, 3.4_

  - [ ] 3.2 Update course service for default assignment
    - Modify get_or_create_settings to use default prompt
    - Ensure existing courses get default prompt via migration
    - _Requirements: 3.2_

  - [ ] 3.3 Write property test for default prompt assignment
    - **Property 2: Default Prompt Assignment**
    - **Validates: Requirements 3.2**

- [ ] 4. Chat Integration Updates
  - [ ] 4.1 Update chat service to use system prompts
    - Modify chat endpoint to include course system prompt
    - Implement fallback to default prompt when custom prompt is empty
    - _Requirements: 4.1, 4.3_

  - [ ] 4.2 Update chat request processing
    - Prepend system prompt to LLM conversation context
    - Ensure prompt is applied to all course chat requests
    - _Requirements: 4.1, 4.4_

  - [ ] 4.3 Write property test for chat integration
    - **Property 4: Chat Integration**
    - **Validates: Requirements 4.1, 4.3**

- [ ] 5. Frontend API Integration
  - [x] 5.1 Update TypeScript interfaces
    - Add system_prompt field to CourseSettings interface
    - Add system_prompt field to CourseSettingsUpdate interface
    - Update API client methods
    - _Requirements: 2.3_

  - [x] 5.2 Update API client methods
    - Ensure getCourseSettings returns system_prompt
    - Ensure updateCourseSettings accepts system_prompt
    - _Requirements: 2.3_

- [ ] 6. Enhanced Settings UI Implementation
  - [x] 6.1 Redesign settings layout structure
    - Create distinct visual sections with proper spacing
    - Implement card-based layout with clear borders
    - Add prominent section headers with descriptions
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 6.2 Implement system prompt editor
    - Add large textarea for system prompt editing
    - Implement character counter with limit warnings
    - Add helpful placeholder text and instructions
    - _Requirements: 2.2, 5.4_

  - [x] 6.3 Add system prompt validation
    - Implement real-time character counting
    - Add warning states at 1900+ characters
    - Add error states at 2000+ characters
    - _Requirements: 5.1, 5.4_

  - [x] 6.4 Write property test for UI validation
    - **Property 3: Character Limit Enforcement**
    - **Validates: Requirements 5.1**

  - [ ] 6.5 Write property test for character counter
    - **Property 5: Settings UI Validation**
    - **Validates: Requirements 5.4**

- [-] 7. Visual Design Improvements
  - [x] 7.1 Enhance section visual separation
    - Add consistent card styling with shadows/borders
    - Implement proper spacing between sections
    - Use consistent color scheme and typography
    - _Requirements: 1.1, 1.4_

  - [ ] 7.2 Improve form field organization
    - Organize related fields in logical grid layouts
    - Add consistent field spacing and alignment
    - Implement responsive design for different screen sizes
    - _Requirements: 1.5_

  - [ ] 7.3 Add loading and error states
    - Implement loading spinners for save operations
    - Add error message displays with retry options
    - Include success feedback for saved changes
    - _Requirements: 5.2_

- [ ] 8. Integration and Testing
  - [ ] 8.1 Test end-to-end functionality
    - Verify system prompt saves and loads correctly
    - Test chat integration with custom prompts
    - Validate UI responsiveness and error handling
    - _Requirements: 2.3, 4.1_

  - [ ] 8.2 Write integration tests
    - Test complete settings save/load cycle
    - Test chat functionality with system prompts
    - Test UI component interactions

- [ ] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases