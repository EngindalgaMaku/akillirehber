# Implementation Plan: Admin User Management

## Overview

This implementation plan adds admin user management functionality to the RAG chatbot system. The user with ID=1 will have admin privileges, displayed with a badge in the sidebar, and access to a user management panel for CRUD operations on all users.

## Tasks

- [x] 1. Backend: Database Models and Migrations
  - Create AuditLog model for tracking admin operations
  - Create TemporaryPassword model for password resets
  - Add is_active column to User model if not exists
  - Add last_login column to User model if not exists
  - Create Alembic migration for new models and columns
  - _Requirements: 1.4, 8.5, 10.5_

- [x] 2. Backend: Admin Authentication Middleware
  - [x] 2.1 Create admin verification utility function
    - Implement `is_admin(user_id: int) -> bool` function
    - Add `get_admin_flag_from_jwt(token: str) -> bool` function
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [ ]* 2.2 Write property test for admin recognition
    - **Property 1: Admin Recognition**
    - **Validates: Requirements 1.1, 1.2**
  
  - [x] 2.3 Create admin-only route dependency
    - Implement `require_admin` FastAPI dependency
    - Verify JWT contains admin flag
    - Return 403 if not admin
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  
  - [ ]* 2.4 Write property test for authorization enforcement
    - **Property 16: Authorization Enforcement**
    - **Validates: Requirements 8.2, 8.4**

- [x] 3. Backend: Audit Logging Service
  - [x] 3.1 Create audit logging service
    - Implement `log_admin_action()` function
    - Store admin_user_id, action, target_user_id, details, timestamp, ip_address
    - _Requirements: 8.5, 8.6_
  
  - [ ]* 3.2 Write property test for audit logging
    - **Property 17: Audit Logging Completeness**
    - **Validates: Requirements 8.5, 8.6**

- [ ] 4. Backend: User Management API Endpoints
  - [x] 4.1 Implement GET /api/admin/users endpoint
    - Support pagination (page, limit)
    - Support role filtering
    - Support search by name/email
    - Support sorting by any column
    - Return user list with all required fields
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7_
  
  - [ ]* 4.2 Write property tests for user list endpoint
    - **Property 5: User List Completeness**
    - **Property 6: Sorting Correctness**
    - **Property 7: Role Filtering**
    - **Property 8: Search Functionality**
    - **Property 9: Pagination Consistency**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.7**
  
  - [x] 4.3 Implement PUT /api/admin/users/{userId} endpoint ✅
    - Validate email uniqueness
    - Prevent editing user with ID=1
    - Update user in database
    - Log admin action
    - _Requirements: 5.3, 5.4, 5.5, 5.7_
  
  - [ ]* 4.4 Write property tests for user update endpoint
    - **Property 10: Email Uniqueness Validation**
    - **Property 11: User Update Persistence**
    - **Property 12: Admin Self-Protection**
    - **Validates: Requirements 5.4, 5.5, 5.7**
  
  - [x] 4.5 Implement DELETE /api/admin/users/{userId} endpoint ✅
    - Prevent deleting user with ID=1
    - Cascade delete related data (courses, documents, chat history)
    - Log admin action
    - _Requirements: 7.3, 7.4, 7.6_
  
  - [ ]* 4.6 Write property tests for user delete endpoint
    - **Property 12: Admin Self-Protection**
    - **Property 15: Cascading Deletion**
    - **Validates: Requirements 5.8, 7.4, 7.6**

- [x] 5. Checkpoint - Backend Core Functionality
  - Ensure all backend tests pass
  - Verify database migrations work correctly
  - Test API endpoints with Postman or curl
  - Ask the user if questions arise



- [x] 6. Backend: User Deactivation Endpoints
  - [x] 6.1 Implement PATCH /api/admin/users/{userId}/deactivate endpoint
    - Prevent deactivating user with ID=1
    - Set is_active to False
    - Log admin action
    - _Requirements: 6.1, 6.6_
  
  - [x] 6.2 Implement PATCH /api/admin/users/{userId}/activate endpoint
    - Set is_active to True
    - Log admin action
    - _Requirements: 6.4_
  
  - [x] 6.3 Update login endpoint to check is_active
    - Reject login if user is inactive
    - Return "Hesabınız devre dışı bırakılmış" message
    - _Requirements: 6.2, 6.3_
  
  - [ ]* 6.4 Write property tests for deactivation
    - **Property 13: Deactivation Round-Trip**
    - **Property 14: Inactive Login Prevention**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

- [x] 7. Backend: Statistics and Password Reset Endpoints
  - [x] 7.1 Implement GET /api/admin/statistics endpoint
    - Calculate total users, active teachers, active students, inactive users
    - Calculate new users this month
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ]* 7.2 Write property test for statistics accuracy
    - **Property 18: Statistics Accuracy**
    - **Validates: Requirements 9.3**
  
  - [x] 7.3 Implement POST /api/admin/users/{userId}/reset-password endpoint ✅
    - Generate temporary password
    - Store in TemporaryPassword table with 24-hour expiration
    - Send email to user
    - Log admin action
    - _Requirements: 10.1, 10.2, 10.3, 10.5, 10.6_
  
  - [ ]* 7.4 Write property tests for password reset
    - **Property 19: Temporary Password Expiration**
    - **Property 20: Password Reset Logging**
    - **Validates: Requirements 10.3, 10.5, 10.6**
  
  - [x] 7.5 Update login endpoint to handle temporary passwords
    - Check if password is temporary
    - Force password change on first login
    - Mark temporary password as used
    - _Requirements: 10.4_

- [x] 8. Backend: JWT Token Enhancement
  - [x] 8.1 Update JWT token generation to include admin flag
    - Add "is_admin" claim to JWT payload when user ID=1
    - _Requirements: 1.3_
  
  - [ ]* 8.2 Write property test for JWT admin flag
    - **Property 2: Admin JWT Flag**
    - **Validates: Requirements 1.3**

- [x] 9. Checkpoint - Backend Complete
  - Ensure all backend tests pass
  - Verify all API endpoints work correctly
  - Test authorization on all admin endpoints
  - Ask the user if questions arise

- [x] 10. Frontend: RoleBadge Component
  - [x] 10.1 Create RoleBadge component
    - Accept role prop (admin, teacher, student)
    - Display appropriate text and styling
    - Admin: Gold/amber background
    - Teacher: Blue background
    - Student: Green background
    - _Requirements: 2.1, 2.2, 2.4, 2.5_
  
  - [ ]* 10.2 Write property test for role badge display
    - **Property 3: Role Badge Display**
    - **Validates: Requirements 2.1, 2.4, 2.5**

- [x] 11. Frontend: Sidebar Enhancement
  - [x] 11.1 Update Sidebar component to display RoleBadge
    - Add RoleBadge next to user name
    - Ensure badge persists across all pages
    - _Requirements: 2.3, 2.6_
  
  - [x] 11.2 Add "Kullanıcı Yönetimi" menu item for admin
    - Show menu item only when user.id === 1
    - Add appropriate icon (Users, Shield, or Settings)
    - Link to /admin/users route
    - _Requirements: 3.1, 3.3, 3.4_
  
  - [ ]* 11.3 Write property test for admin menu visibility
    - **Property 4: Admin Menu Visibility**
    - **Validates: Requirements 3.1, 3.4**

- [x] 12. Frontend: User Management Panel - User List
  - [x] 12.1 Create UserManagementPanel component
    - Fetch users from /api/admin/users
    - Display user table with all required columns
    - Implement column sorting
    - Implement role filtering dropdown
    - Implement search input
    - Implement pagination controls
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7_
  
  - [x] 12.2 Handle empty user list state
    - Display "Henüz kullanıcı yok" message when list is empty
    - _Requirements: 4.6_
  
  - [x] 12.3 Add visual distinction for inactive users
    - Gray out or strikethrough inactive users
    - _Requirements: 6.5_

- [x] 13. Frontend: User Management Panel - Statistics Dashboard
  - [x] 13.1 Create StatisticsDashboard component
    - Fetch statistics from /api/admin/statistics
    - Display total users, active teachers, active students, inactive users, new users this month
    - Use cards or charts for visualization
    - Update statistics when users are added/removed
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 14. Checkpoint - Frontend Core UI
  - Ensure sidebar displays correctly for all roles
  - Verify user list displays and functions correctly
  - Test sorting, filtering, and search
  - Ask the user if questions arise



- [x] 15. Frontend: User Edit Dialog
  - [x] 15.1 Create UserEditDialog component
    - Display dialog when "Düzenle" button is clicked
    - Show form with name, email, role, and active status fields
    - Validate email format
    - Call PUT /api/admin/users/{userId} on save
    - Display success message on successful update
    - Display validation errors inline
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6_
  
  - [x] 15.2 Prevent editing user with ID=1
    - Disable edit button for user with ID=1
    - Or show read-only dialog
    - _Requirements: 5.7_

- [ ] 16. Frontend: User Delete Confirmation
  - [ ] 16.1 Create DeleteConfirmationDialog component
    - Display dialog when "Sil" button is clicked
    - Show warning about data loss
    - List what will be deleted (courses, documents, chat history)
    - Call DELETE /api/admin/users/{userId} on confirm
    - Display success message on successful deletion
    - Display error message if deletion fails
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.7_
  
  - [ ] 16.2 Prevent deleting user with ID=1
    - Disable delete button for user with ID=1
    - _Requirements: 7.6_

- [ ] 17. Frontend: User Deactivation Toggle
  - [ ] 17.1 Add deactivate/activate toggle button
    - Show "Deaktif Et" button for active users
    - Show "Aktif Et" button for inactive users
    - Call PATCH /api/admin/users/{userId}/deactivate or activate
    - Update user list after successful operation
    - _Requirements: 6.1, 6.4_
  
  - [ ] 17.2 Prevent deactivating user with ID=1
    - Disable deactivate button for user with ID=1
    - _Requirements: 6.6_

- [ ] 18. Frontend: Password Reset Dialog
  - [ ] 18.1 Create PasswordResetDialog component
    - Display dialog when "Şifre Sıfırla" button is clicked
    - Call POST /api/admin/users/{userId}/reset-password
    - Display temporary password to admin
    - Show expiration time (24 hours)
    - Show confirmation that email was sent
    - _Requirements: 10.1, 10.2, 10.3_

- [ ] 19. Frontend: Error Handling and Loading States
  - [ ] 19.1 Add loading spinners for async operations
    - Show spinner while fetching user list
    - Show spinner while updating/deleting users
    - _Requirements: 7.7_
  
  - [ ] 19.2 Add error toast notifications
    - Display network errors with retry option
    - Display authorization errors (redirect to login)
    - Display server errors with error code
    - _Requirements: 7.7_

- [ ] 20. Frontend: Authorization Guard
  - [ ] 20.1 Create admin route guard
    - Check if user.id === 1 before rendering admin pages
    - Redirect to home page if not admin
    - Display "Yetkisiz erişim" message
    - _Requirements: 8.1, 8.2_

- [ ] 21. Checkpoint - Frontend Complete
  - Ensure all dialogs work correctly
  - Test all CRUD operations end-to-end
  - Verify error handling works
  - Ask the user if questions arise

- [ ] 22. Integration Testing
  - [ ]* 22.1 Write integration test for admin login flow
    - Login as admin → Verify badge → Access user management → Verify list
  
  - [ ]* 22.2 Write integration test for user edit flow
    - Login as admin → Edit user → Verify database update
  
  - [ ]* 22.3 Write integration test for user delete flow
    - Login as admin → Delete user → Verify cascading deletion
  
  - [ ]* 22.4 Write integration test for unauthorized access
    - Login as teacher → Attempt admin access → Verify 403 error
  
  - [ ]* 22.5 Write integration test for password reset flow
    - Login as admin → Reset password → Verify email → Login with temp password

- [ ] 23. Security Testing
  - [ ]* 23.1 Test SQL injection prevention
    - Attempt SQL injection in search and filter inputs
  
  - [ ]* 23.2 Test XSS prevention
    - Attempt XSS in user input fields
  
  - [ ]* 23.3 Test unauthorized access logging
    - Attempt unauthorized access and verify audit log

- [ ] 24. Final Checkpoint
  - Ensure all tests pass (unit, property, integration)
  - Verify all requirements are met
  - Test with real data in development environment
  - Ask the user if ready for deployment

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Backend uses FastAPI with SQLAlchemy ORM and Alembic migrations
- Frontend uses Next.js with TypeScript and React components
- Testing frameworks: Hypothesis (Python), fast-check (TypeScript)
