# Requirements Document

## Introduction

Sistemde ID'si 1 olan öğretmen hesabı Yönetici (Admin) yetkisine sahiptir. Bu yönetici, tüm kullanıcıları görüntüleyebilir, düzenleyebilir ve yönetebilir. Sidebar'da yönetici yetkisi görsel olarak vurgulanmalı ve yönetici paneline erişim sağlanmalıdır.

## Glossary

- **Admin_User**: ID'si 1 olan, tüm kullanıcıları yönetme yetkisine sahip öğretmen hesabı
- **Teacher_User**: Normal öğretmen hesabı, sadece kendi derslerini yönetebilir
- **Student_User**: Öğrenci hesabı, derslere erişip soru sorabilir
- **User_Management_Panel**: Yöneticinin kullanıcıları görüntüleyip düzenleyebildiği panel
- **Sidebar**: Sol taraftaki navigasyon menüsü
- **Role_Badge**: Kullanıcı rolünü gösteren görsel etiket

## Requirements

### Requirement 1: Yönetici Rolü Tanımlama

**User Story:** As a system, I want to identify the admin user by ID, so that special privileges can be granted.

#### Acceptance Criteria

1. THE System SHALL recognize user with ID=1 as Admin_User
2. WHEN a user logs in, THE System SHALL check if user ID equals 1
3. IF user ID equals 1, THEN THE System SHALL set admin flag in JWT token
4. THE System SHALL store admin status in user session
5. WHEN admin status changes, THE System SHALL update the session immediately

### Requirement 2: Sidebar Yönetici Vurgusu

**User Story:** As an admin, I want to see my admin status in the sidebar, so that I know I have special privileges.

#### Acceptance Criteria

1. WHEN Admin_User views the sidebar, THE System SHALL display a Role_Badge with "Yönetici" text
2. THE Role_Badge SHALL use a distinct color (e.g., gold, red, or purple) to stand out
3. THE Role_Badge SHALL appear next to or below the user's name in the sidebar
4. WHEN a Teacher_User views the sidebar, THE System SHALL display "Öğretmen" badge
5. WHEN a Student_User views the sidebar, THE System SHALL display "Öğrenci" badge
6. THE Role_Badge SHALL be visible on all pages while sidebar is open

### Requirement 3: Yönetici Menü Öğesi

**User Story:** As an admin, I want a dedicated menu item in the sidebar, so that I can access user management features.

#### Acceptance Criteria

1. WHEN Admin_User views the sidebar, THE System SHALL display "Kullanıcı Yönetimi" menu item
2. THE "Kullanıcı Yönetimi" menu item SHALL have an appropriate icon (e.g., users, shield, settings)
3. WHEN Admin_User clicks "Kullanıcı Yönetimi", THE System SHALL navigate to User_Management_Panel
4. WHEN Teacher_User or Student_User views the sidebar, THE System SHALL NOT display "Kullanıcı Yönetimi" menu item
5. THE "Kullanıcı Yönetimi" menu item SHALL be positioned prominently in the sidebar

### Requirement 4: Kullanıcı Listeleme

**User Story:** As an admin, I want to view all users in the system, so that I can manage them effectively.

#### Acceptance Criteria

1. WHEN Admin_User accesses User_Management_Panel, THE System SHALL display a list of all users
2. THE user list SHALL display the following information for each user:
   - User ID
   - Full name
   - Email address
   - Role (Admin/Teacher/Student)
   - Account creation date
   - Last login date
3. THE user list SHALL support sorting by any column
4. THE user list SHALL support filtering by role
5. THE user list SHALL support search by name or email
6. WHEN the user list is empty, THE System SHALL display "Henüz kullanıcı yok" message
7. THE user list SHALL paginate results if more than 50 users exist

### Requirement 5: Kullanıcı Düzenleme

**User Story:** As an admin, I want to edit user information, so that I can correct errors or update details.

#### Acceptance Criteria

1. WHEN Admin_User clicks "Düzenle" button on a user, THE System SHALL open an edit dialog
2. THE edit dialog SHALL allow editing the following fields:
   - Full name
   - Email address
   - Role (Teacher/Student only, cannot change admin)
   - Active/Inactive status
3. WHEN Admin_User saves changes, THE System SHALL validate all fields
4. IF email is already used by another user, THEN THE System SHALL display validation error
5. WHEN changes are saved successfully, THE System SHALL update the user in database
6. WHEN changes are saved successfully, THE System SHALL display success message
7. THE System SHALL NOT allow Admin_User to edit their own admin status
8. THE System SHALL NOT allow Admin_User to delete user with ID=1

### Requirement 6: Kullanıcı Deaktivasyonu

**User Story:** As an admin, I want to deactivate users, so that I can prevent access without deleting accounts.

#### Acceptance Criteria

1. WHEN Admin_User clicks "Deaktif Et" button on a user, THE System SHALL mark user as inactive
2. WHEN a user is inactive, THE System SHALL prevent login attempts
3. IF an inactive user tries to login, THEN THE System SHALL display "Hesabınız devre dışı bırakılmış" message
4. WHEN Admin_User clicks "Aktif Et" button on an inactive user, THE System SHALL reactivate the account
5. THE user list SHALL visually distinguish inactive users (e.g., grayed out, strikethrough)
6. THE System SHALL NOT allow deactivating user with ID=1

### Requirement 7: Kullanıcı Silme

**User Story:** As an admin, I want to delete users, so that I can remove accounts that are no longer needed.

#### Acceptance Criteria

1. WHEN Admin_User clicks "Sil" button on a user, THE System SHALL display confirmation dialog
2. THE confirmation dialog SHALL warn about data loss
3. WHEN Admin_User confirms deletion, THE System SHALL remove user from database
4. WHEN a user is deleted, THE System SHALL also delete:
   - User's courses (if teacher)
   - User's chat history
   - User's uploaded documents
5. WHEN deletion is successful, THE System SHALL display success message
6. THE System SHALL NOT allow deleting user with ID=1
7. IF deletion fails, THEN THE System SHALL display error message with reason

### Requirement 8: Yetki Kontrolü

**User Story:** As a system, I want to verify admin privileges on all admin operations, so that security is maintained.

#### Acceptance Criteria

1. WHEN a user attempts to access User_Management_Panel, THE System SHALL verify admin status
2. IF user is not Admin_User, THEN THE System SHALL return 403 Forbidden error
3. WHEN a user attempts admin API endpoints, THE System SHALL verify JWT token contains admin flag
4. IF JWT token does not contain admin flag, THEN THE System SHALL reject the request
5. THE System SHALL log all admin operations with timestamp and admin user ID
6. WHEN an unauthorized access attempt occurs, THE System SHALL log the security event

### Requirement 9: Kullanıcı İstatistikleri

**User Story:** As an admin, I want to see user statistics, so that I can understand system usage.

#### Acceptance Criteria

1. WHEN Admin_User accesses User_Management_Panel, THE System SHALL display statistics dashboard
2. THE dashboard SHALL show:
   - Total number of users
   - Number of active teachers
   - Number of active students
   - Number of inactive users
   - New users this month
3. THE statistics SHALL update in real-time when users are added/removed
4. THE dashboard SHALL display statistics using charts or cards

### Requirement 10: Şifre Sıfırlama

**User Story:** As an admin, I want to reset user passwords, so that I can help users who forgot their passwords.

#### Acceptance Criteria

1. WHEN Admin_User clicks "Şifre Sıfırla" button on a user, THE System SHALL generate a temporary password
2. THE System SHALL display the temporary password to Admin_User
3. THE System SHALL send the temporary password to user's email
4. WHEN user logs in with temporary password, THE System SHALL force password change
5. THE temporary password SHALL expire after 24 hours
6. THE System SHALL log all password reset operations
