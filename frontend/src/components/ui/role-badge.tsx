import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Shield, GraduationCap, BookOpen } from "lucide-react"

import { cn } from "@/lib/utils"

const roleBadgeVariants = cva(
  "inline-flex items-center justify-center rounded-full w-5 h-5",
  {
    variants: {
      role: {
        admin: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
        teacher: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
        student: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
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

const roleIconMap = {
  admin: Shield,
  teacher: GraduationCap,
  student: BookOpen,
} as const

const roleTextMap = {
  admin: "Yönetici",
  teacher: "Öğretmen",
  student: "Öğrenci",
} as const

function RoleBadge({ className, role, ...props }: Readonly<RoleBadgeProps>) {
  const Icon = roleIconMap[role]
  
  return (
    <span
      className={cn(roleBadgeVariants({ role, className }))}
      title={roleTextMap[role]}
      {...props}
    >
      <Icon className="w-3 h-3" />
    </span>
  )
}

export { RoleBadge, roleBadgeVariants }
