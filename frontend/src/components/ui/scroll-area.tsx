import * as React from "react"
import { cn } from "@/lib/utils"

interface ScrollAreaProps {
  children: React.ReactNode
  className?: string
}

const ScrollArea: React.FC<ScrollAreaProps> = ({ children, className }) => {
  return (
    <div className={cn("relative overflow-auto", className)}>
      {children}
    </div>
  )
}

export { ScrollArea }
