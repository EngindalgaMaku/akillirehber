# RoleBadge Component

## Overview
A reusable badge component for displaying user roles with appropriate styling.

## Usage

```tsx
import { RoleBadge } from "@/components/ui/role-badge"

// Admin badge (gold/amber)
<RoleBadge role="admin" />

// Teacher badge (blue)
<RoleBadge role="teacher" />

// Student badge (green)
<RoleBadge role="student" />

// With custom className
<RoleBadge role="admin" className="ml-2" />
```

## Props

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| role | "admin" \| "teacher" \| "student" | Yes | The user role to display |
| className | string | No | Additional CSS classes |

## Styling

- **Admin**: Gold/amber background (`bg-amber-100`) with dark amber text (`text-amber-800`)
- **Teacher**: Blue background (`bg-blue-100`) with dark blue text (`text-blue-800`)
- **Student**: Green background (`bg-green-100`) with dark green text (`text-green-800`)

All badges include:
- Rounded full shape
- Small text size (text-xs)
- Medium font weight
- Padding (px-2.5 py-0.5)
- Border matching the background color
- Dark mode support

## Text Display

The component automatically displays the correct Turkish text:
- `admin` → "Yönetici"
- `teacher` → "Öğretmen"
- `student` → "Öğrenci"

## Requirements Satisfied

- ✅ 2.1: Display role badge with appropriate text
- ✅ 2.2: Use distinct colors for each role
- ✅ 2.4: Display "Yönetici" for admin
- ✅ 2.5: Display "Öğretmen" for teacher and "Öğrenci" for student
