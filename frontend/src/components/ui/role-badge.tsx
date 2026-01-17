import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const roleBadgeVariants = cva(
  "inline-flex items-center justify-center rounded-full text-xs font-medium px-2.5 py-0.5",
  {
    variants: {
      role: {
        admin: "bg-amber-100 text-amber-800 border border-amber-200 dark:bg-amber-900/30 dark:text-amber-400 dark:border-amber-800",
        teacher: "bg-blue-100 text-blue-800 border border-blue-200 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-800",
        student: "bg-green-100 text-green-800 border border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-800",
      },
    },
    defaultVariants: {
      role: "student",
    },
  }
)

export interface RoleBadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof roleBadgeVariants> {
  role: "admin" | "teacher" | "student"
}

const roleTextMap = {
  admin: "Yönetici",
  teacher: "Öğretmen",
  student: "Öğrenci",
} as const

function RoleBadge({ className, role, ...props }: Readonly<RoleBadgeProps>) {
  return (
    <span
      className={cn(roleBadgeVariants({ role, className }))}
      {...props}
    >
      {roleTextMap[role]}
    </span>
  )
}

export { RoleBadge, roleBadgeVariants }
