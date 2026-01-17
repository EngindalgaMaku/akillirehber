# Implementation Plan: Student Portal

## Overview

Bu plan, öğrenci portalı özelliğini mevcut Next.js frontend yapısına entegre edecektir. Öğrenciler için ayrı bir `/student` route yapısı oluşturulacak ve rol bazlı navigasyon sağlanacaktır.

## Tasks

- [x] 1. Student Layout ve Routing Yapısı
  - [x] 1.1 Create student layout component with simplified sidebar
    - Create `frontend/src/app/student/layout.tsx`
    - Include only Dersler, Profil, and Çıkış navigation items
    - Reuse existing UI components (Button, icons)
    - Add role check to redirect teachers to dashboard
    - _Requirements: 1.1, 1.2, 6.1_

  - [x] 1.2 Update auth context for role-based routing
    - Modify login redirect logic in `auth-context.tsx`
    - Students redirect to `/student`, teachers to `/dashboard`
    - _Requirements: 1.3_

- [x] 2. Course Selection Page
  - [x] 2.1 Create student courses page
    - Create `frontend/src/app/student/page.tsx`
    - Display courses in card format
    - Show course name and description
    - Click navigates to `/student/chat/[courseId]`
    - No create/edit/delete buttons for students
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. Student Chat Page
  - [x] 3.1 Create student chat page with full-screen interface
    - Create `frontend/src/app/student/chat/[courseId]/page.tsx`
    - Reuse ChatTab component logic
    - Add back button to course selection
    - Display course name in header
    - _Requirements: 3.1, 3.2, 3.6_

  - [x] 3.2 Implement source references display
    - Show source references in chat responses
    - Modal for full source content on click
    - Display document name and relevance score
    - _Requirements: 3.3, 3.4_

  - [x] 3.3 Add loading states and error handling
    - Loading indicator during AI response
    - Error messages in Turkish
    - _Requirements: 3.5_

- [x] 4. Chat History Implementation
  - [x] 4.1 Create chat history service
    - Create `frontend/src/lib/chat-history.ts`
    - Implement getHistory, saveHistory, clearHistory functions
    - Use localStorage with course-specific keys
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 4.2 Write property test for chat history round-trip
    - **Property 2: Chat History Persistence Round-trip**
    - **Validates: Requirements 4.1, 4.2**

  - [ ]* 4.3 Write property test for course history isolation
    - **Property 3: Course-specific History Isolation**
    - **Validates: Requirements 4.3**

  - [x] 4.4 Integrate chat history with chat page
    - Load history on page mount
    - Save after each message exchange
    - Clear button functionality
    - Display timestamps
    - _Requirements: 4.4, 4.5_

- [x] 5. Profile Page
  - [x] 5.1 Create student profile page
    - Create `frontend/src/app/student/profile/page.tsx`
    - Display full name, email, role
    - Edit form for name and email
    - Password change form
    - Success/error messages
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 6. Navigation and Polish
  - [x] 6.1 Implement active page highlighting
    - Highlight current page in sidebar
    - _Requirements: 6.3_

  - [x] 6.2 Implement logout functionality
    - Clear session and redirect to login
    - _Requirements: 6.4_

  - [x] 6.3 Add back navigation from chat
    - Back button in chat header
    - _Requirements: 6.2_

- [x] 7. Checkpoint - Ensure all tests pass ✓
  - Backend: 52 tests passed
  - Frontend: All lint issues resolved (0 errors, 0 warnings)

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Reuse existing components where possible (ChatTab logic, UI components)
