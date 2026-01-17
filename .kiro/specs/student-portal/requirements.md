# Requirements Document

## Introduction

Bu özellik, öğrenciler için basitleştirilmiş bir portal oluşturmayı amaçlamaktadır. Öğrenci portalı, öğrencilerin sadece ders seçimi yapabilmesi, seçilen ders hakkında AI asistanı ile sohbet edebilmesi ve sohbet geçmişini görüntüleyebilmesi için tasarlanmıştır. Ayrıca öğrencilerin profil bilgilerini görüntüleyip düzenleyebileceği bir profil sayfası bulunacaktır.

## Glossary

- **Student_Portal**: Öğrencilere özel basitleştirilmiş arayüz
- **Course_Selector**: Öğrencinin erişebileceği dersleri listeleyen ve seçim yapmasını sağlayan bileşen
- **Chat_Interface**: Seçilen ders hakkında AI asistanı ile sohbet yapılabilen ekran
- **Chat_History**: Öğrencinin geçmiş sohbetlerini saklayan ve görüntüleyen sistem
- **Profile_Page**: Öğrencinin kişisel bilgilerini görüntüleyip düzenleyebildiği sayfa
- **Student**: Öğrenci rolüne sahip kullanıcı

## Requirements

### Requirement 1: Öğrenci Portalı Ana Yapısı

**User Story:** As a student, I want a simplified portal interface, so that I can easily access course materials and chat with the AI assistant without unnecessary complexity.

#### Acceptance Criteria

1. WHEN a student logs in, THE Student_Portal SHALL display a simplified sidebar with only essential navigation items (Dersler, Profil, Çıkış)
2. THE Student_Portal SHALL NOT display admin-only features such as Chunking, RAGAS, or System Settings
3. WHEN a student accesses the portal, THE Student_Portal SHALL redirect to the course selection page as the default landing page

### Requirement 2: Ders Seçimi

**User Story:** As a student, I want to see and select available courses, so that I can access course materials and chat about them.

#### Acceptance Criteria

1. WHEN a student visits the courses page, THE Course_Selector SHALL display all active courses in a card or list format
2. WHEN a student clicks on a course, THE Course_Selector SHALL navigate to the chat interface for that course
3. THE Course_Selector SHALL display course name and description for each available course
4. THE Course_Selector SHALL NOT allow students to create, edit, or delete courses

### Requirement 3: Chat Arayüzü

**User Story:** As a student, I want to chat with an AI assistant about course materials, so that I can get help understanding the content.

#### Acceptance Criteria

1. WHEN a student selects a course, THE Chat_Interface SHALL display a full-screen chat interface
2. WHEN a student sends a message, THE Chat_Interface SHALL send the message to the AI and display the response
3. THE Chat_Interface SHALL display source references when the AI uses course materials to answer
4. WHEN a student clicks on a source reference, THE Chat_Interface SHALL display the full content of that source
5. THE Chat_Interface SHALL display a loading indicator while waiting for AI response
6. THE Chat_Interface SHALL allow students to navigate back to course selection

### Requirement 4: Sohbet Geçmişi

**User Story:** As a student, I want my chat history to be saved, so that I can continue conversations and review past interactions.

#### Acceptance Criteria

1. WHEN a student sends messages in a course chat, THE Chat_History SHALL persist the conversation in local storage
2. WHEN a student returns to a course chat, THE Chat_History SHALL restore the previous conversation
3. THE Chat_History SHALL store conversations separately for each course
4. WHEN a student clicks the clear button, THE Chat_History SHALL delete the conversation history for that course
5. THE Chat_History SHALL display timestamps for each message

### Requirement 5: Profil Sayfası

**User Story:** As a student, I want to view and edit my profile information, so that I can keep my account details up to date.

#### Acceptance Criteria

1. WHEN a student visits the profile page, THE Profile_Page SHALL display the student's full name, email, and role
2. WHEN a student edits their profile, THE Profile_Page SHALL allow updating full name and email
3. WHEN a student submits profile changes, THE Profile_Page SHALL save the changes and display a success message
4. THE Profile_Page SHALL allow students to change their password
5. IF a profile update fails, THEN THE Profile_Page SHALL display an appropriate error message

### Requirement 6: Navigasyon ve Yönlendirme

**User Story:** As a student, I want clear navigation between pages, so that I can easily move around the portal.

#### Acceptance Criteria

1. THE Student_Portal SHALL provide a consistent sidebar navigation on all pages
2. WHEN a student is on the chat page, THE Student_Portal SHALL display a back button to return to course selection
3. THE Student_Portal SHALL highlight the current active page in the navigation
4. WHEN a student logs out, THE Student_Portal SHALL redirect to the login page and clear the session
