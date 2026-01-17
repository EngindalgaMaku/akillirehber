# Design Document

## Overview

This design implements an admin user management system where the user with ID=1 has special administrative privileges. The admin can view, edit, deactivate, and delete users through a dedicated management panel. The admin role is visually highlighted in the sidebar with a badge, and a special "Kullanıcı Yönetimi" menu item provides access to the management interface.

## Architecture

### System Components

1. **Frontend (Next.js)**
   - Sidebar component with role badge display
   - User management panel with CRUD operations
   - Admin-only menu items and routes
   - Role-based UI rendering

2. **Backend (FastAPI)**
   - Admin authentication middleware
   - User management API endpoints
   - Role verification logic
   - Audit logging system

3. **Database (PostgreSQL)**
   - User table with admin flag
   - Audit log table for admin operations
   - User activity tracking

### Authentication Flow

```
User Login → Check ID=1 → Set admin flag in JWT → Store in session → Render UI based on role
```

### Authorization Flow

```
API Request → Extract JWT → Verify admin flag → Allow/Deny → Log operation
```

## Components and Interfaces

### Frontend Components

#### 1. Sidebar Component Enhancement

**Location**: `frontend/src/components/Sidebar.tsx`

**Props**:
```typescript
interface SidebarProps {
  user: {
    id: number;
    name: string;
    email: string;
    role: 'admin' | 'teacher' | 'student';
  };
  isOpen: boolean;
}
```

**Behavior**:
- Display role badge based on user role
- Show "Kullanıcı Yönetimi" menu item only for admin
- Highlight active menu item
- Persist badge visibility across all pages

#### 2. RoleBadge Component

**Location**: `frontend/src/components/RoleBadge.tsx`

**Props**:
```typescript
interface RoleBadgeProps {
  role: 'admin' | 'teacher' | 'student';
  className?: string;
}
```

**Styling**:
- Admin: Gold/amber background with dark text
- Teacher: Blue background with white text
- Student: Green background with white text

#### 3. UserManagementPanel Component

**Location**: `frontend/src/components/admin/UserManagementPanel.tsx`

**Features**:
- User list with sorting, filtering, searching
- Statistics dashboard
- Edit user dialog
- Delete confirmation dialog
- Deactivate/activate toggle
- Password reset functionality



### Backend API Endpoints

#### 1. Admin Verification Endpoint

```
GET /api/auth/verify-admin
Response: { "isAdmin": boolean, "userId": number }
```

#### 2. User List Endpoint

```
GET /api/admin/users?page=1&limit=50&role=teacher&search=john
Headers: Authorization: Bearer <JWT>
Response: {
  "users": [
    {
      "id": number,
      "name": string,
      "email": string,
      "role": string,
      "isActive": boolean,
      "createdAt": string,
      "lastLogin": string
    }
  ],
  "total": number,
  "page": number,
  "totalPages": number
}
```

#### 3. User Update Endpoint

```
PUT /api/admin/users/{userId}
Headers: Authorization: Bearer <JWT>
Body: {
  "name": string,
  "email": string,
  "role": "teacher" | "student",
  "isActive": boolean
}
Response: { "success": boolean, "user": User }
```

#### 4. User Delete Endpoint

```
DELETE /api/admin/users/{userId}
Headers: Authorization: Bearer <JWT>
Response: { "success": boolean, "message": string }
```

#### 5. User Statistics Endpoint

```
GET /api/admin/statistics
Headers: Authorization: Bearer <JWT>
Response: {
  "totalUsers": number,
  "activeTeachers": number,
  "activeStudents": number,
  "inactiveUsers": number,
  "newUsersThisMonth": number
}
```

#### 6. Password Reset Endpoint

```
POST /api/admin/users/{userId}/reset-password
Headers: Authorization: Bearer <JWT>
Response: {
  "success": boolean,
  "temporaryPassword": string,
  "expiresAt": string
}
```



## Data Models

### User Model Enhancement

```python
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum('admin', 'teacher', 'student'), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    @property
    def is_admin(self) -> bool:
        return self.id == 1
```

### AuditLog Model

```python
class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    admin_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)  # 'edit', 'delete', 'deactivate', 'reset_password'
    target_user_id = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String, nullable=True)
```

### TemporaryPassword Model

```python
class TemporaryPassword(Base):
    __tablename__ = "temporary_passwords"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Admin Recognition

*For any* user in the system, the user should be recognized as admin if and only if their ID equals 1.

**Validates: Requirements 1.1, 1.2**

### Property 2: Admin JWT Flag

*For any* login attempt, if the user ID equals 1, then the generated JWT token must contain an admin flag set to true.

**Validates: Requirements 1.3**

### Property 3: Role Badge Display

*For any* user role (admin, teacher, student), the sidebar must display the corresponding role badge with the correct text ("Yönetici", "Öğretmen", "Öğrenci").

**Validates: Requirements 2.1, 2.4, 2.5**

### Property 4: Admin Menu Visibility

*For any* user, the "Kullanıcı Yönetimi" menu item should be visible in the sidebar if and only if the user is admin (ID=1).

**Validates: Requirements 3.1, 3.4**

### Property 5: User List Completeness

*For any* request to the user list endpoint by an admin, the response must include all users in the database with all required fields (ID, name, email, role, creation date, last login).

**Validates: Requirements 4.1, 4.2**

### Property 6: Sorting Correctness

*For any* column in the user list, sorting by that column must produce results in the correct ascending or descending order.

**Validates: Requirements 4.3**

### Property 7: Role Filtering

*For any* role filter applied to the user list, all returned users must have the specified role.

**Validates: Requirements 4.4**

### Property 8: Search Functionality

*For any* search term, all returned users must have either their name or email containing the search term (case-insensitive).

**Validates: Requirements 4.5**



### Property 9: Pagination Consistency

*For any* user list with more than 50 users, the pagination must ensure that all users appear exactly once across all pages, with no duplicates or missing users.

**Validates: Requirements 4.7**

### Property 10: Email Uniqueness Validation

*For any* user edit operation, if the new email is already used by another user, the system must reject the update with a validation error.

**Validates: Requirements 5.4**

### Property 11: User Update Persistence

*For any* successful user edit operation, querying the user immediately after must return the updated values.

**Validates: Requirements 5.5**

### Property 12: Admin Self-Protection

*For any* attempt to edit or delete the user with ID=1, the system must reject the operation.

**Validates: Requirements 5.7, 5.8, 6.6, 7.6**

### Property 13: Deactivation Round-Trip

*For any* active user, deactivating then reactivating the user must restore the account to its original active state.

**Validates: Requirements 6.1, 6.4**

### Property 14: Inactive Login Prevention

*For any* inactive user, login attempts must be rejected with the message "Hesabınız devre dışı bırakılmış".

**Validates: Requirements 6.2, 6.3**

### Property 15: Cascading Deletion

*For any* user deletion, all related data (courses if teacher, chat history, uploaded documents) must also be deleted from the database.

**Validates: Requirements 7.4**

### Property 16: Authorization Enforcement

*For any* request to admin endpoints, if the JWT token does not contain an admin flag, the system must return a 403 Forbidden error.

**Validates: Requirements 8.2, 8.4**



### Property 17: Audit Logging Completeness

*For any* admin operation (edit, delete, deactivate, password reset), an audit log entry must be created with timestamp, admin user ID, action type, and target user ID.

**Validates: Requirements 8.5, 8.6**

### Property 18: Statistics Accuracy

*For any* request to the statistics endpoint, the returned counts must match the actual counts in the database (total users, active teachers, active students, inactive users).

**Validates: Requirements 9.3**

### Property 19: Temporary Password Expiration

*For any* temporary password, login attempts after 24 hours from creation must be rejected.

**Validates: Requirements 10.5**

### Property 20: Password Reset Logging

*For any* password reset operation, an audit log entry must be created and an email must be sent to the user.

**Validates: Requirements 10.3, 10.6**

## Error Handling

### Frontend Error Handling

1. **Network Errors**: Display toast notification with retry option
2. **Validation Errors**: Show inline error messages on form fields
3. **Authorization Errors**: Redirect to login page with error message
4. **Server Errors**: Display user-friendly error message with error code

### Backend Error Handling

1. **Invalid User ID**: Return 404 Not Found
2. **Unauthorized Access**: Return 403 Forbidden with audit log
3. **Duplicate Email**: Return 400 Bad Request with validation details
4. **Database Errors**: Return 500 Internal Server Error with logged details
5. **Admin Self-Edit**: Return 400 Bad Request with specific message



## Testing Strategy

### Unit Tests

Unit tests will verify specific examples and edge cases:

1. **Admin Recognition**: Test that user with ID=1 is recognized as admin
2. **JWT Token Generation**: Test admin flag is set correctly in tokens
3. **Role Badge Rendering**: Test correct badge text for each role
4. **Email Validation**: Test duplicate email detection
5. **Empty User List**: Test "Henüz kullanıcı yok" message display
6. **Confirmation Dialogs**: Test delete confirmation dialog appears
7. **Success Messages**: Test success messages appear after operations

### Property-Based Tests

Property-based tests will verify universal properties across all inputs. Each test will run a minimum of 100 iterations with randomly generated data.

**Testing Framework**: Hypothesis (Python backend), fast-check (TypeScript frontend)

**Property Test Examples**:

1. **Property 1 Test**: Generate random users with various IDs, verify only ID=1 is admin
   - **Feature: admin-user-management, Property 1: Admin Recognition**

2. **Property 3 Test**: Generate random user roles, verify correct badge text is displayed
   - **Feature: admin-user-management, Property 3: Role Badge Display**

3. **Property 5 Test**: Generate random user databases, verify all users appear in list
   - **Feature: admin-user-management, Property 5: User List Completeness**

4. **Property 6 Test**: Generate random user lists, sort by each column, verify order
   - **Feature: admin-user-management, Property 6: Sorting Correctness**

5. **Property 8 Test**: Generate random search terms, verify all results match
   - **Feature: admin-user-management, Property 8: Search Functionality**

6. **Property 10 Test**: Generate random email updates, verify duplicates are rejected
   - **Feature: admin-user-management, Property 10: Email Uniqueness Validation**

7. **Property 13 Test**: Generate random users, deactivate then reactivate, verify state
   - **Feature: admin-user-management, Property 13: Deactivation Round-Trip**

8. **Property 16 Test**: Generate random JWT tokens, verify authorization enforcement
   - **Feature: admin-user-management, Property 16: Authorization Enforcement**

### Integration Tests

Integration tests will verify end-to-end workflows:

1. Admin login → View user list → Edit user → Verify database update
2. Admin login → Delete user → Verify cascading deletion
3. Admin login → Reset password → Verify email sent → User login with temp password
4. Non-admin login → Attempt admin access → Verify 403 error
5. Admin deactivate user → User login attempt → Verify rejection

### Security Tests

1. Test unauthorized access attempts are logged
2. Test admin operations require valid JWT with admin flag
3. Test user with ID=1 cannot be deleted or edited
4. Test SQL injection attempts are prevented
5. Test XSS attempts in user input fields are sanitized
