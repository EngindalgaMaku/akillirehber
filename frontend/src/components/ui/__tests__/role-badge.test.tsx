/**
 * Manual verification tests for RoleBadge component
 * 
 * To verify this component works correctly:
 * 1. Import it in a page: import { RoleBadge } from "@/components/ui/role-badge"
 * 2. Use it with different roles:
 *    <RoleBadge role="admin" />
 *    <RoleBadge role="teacher" />
 *    <RoleBadge role="student" />
 * 
 * Expected behavior:
 * - Admin badge: Gold/amber background with "Yönetici" text
 * - Teacher badge: Blue background with "Öğretmen" text
 * - Student badge: Green background with "Öğrenci" text
 */

import { RoleBadge } from "../role-badge"

// Type checking tests
const adminBadge = <RoleBadge role="admin" />
const teacherBadge = <RoleBadge role="teacher" />
const studentBadge = <RoleBadge role="student" />

// With custom className
const customBadge = <RoleBadge role="admin" className="ml-2" />

// Verify all role types are accepted
type RoleType = "admin" | "teacher" | "student"
const roles: RoleType[] = ["admin", "teacher", "student"]
const badges = roles.map(role => <RoleBadge key={role} role={role} />)

export { adminBadge, teacherBadge, studentBadge, customBadge, badges }
