import * as React from 'react'
import { cn } from '@/base/utils'
import eyeOpenIcon from '@/assets/icon_preview-open@2x.png'
import eyeCloseIcon from '@/assets/icon_preview-close-one@2x.png'

interface PasswordInputProps extends Omit<React.ComponentProps<'input'>, 'type'> {
  className?: string
  placeholder?: string
  disabled?: boolean
  error?: boolean
}

function PasswordInput({ className, placeholder, disabled, error, ...props }: PasswordInputProps) {
  const [showPassword, setShowPassword] = React.useState(false)

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword)
  }

  return (
    <div className="relative">
      <input
        type={showPassword ? 'text' : 'password'}
        data-slot="input"
        className={cn(
          'file:text-foreground placeholder:text-muted-foreground selection:bg-primary selection:text-primary-foreground dark:bg-input/10 border-input flex h-[48px] w-full min-w-0 rounded-xl border bg-transparent px-3 py-1 text-base shadow-xs transition-[color,box-shadow] outline-none file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
          'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
          'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
          'border-none placeholder-gray-400',
          'pr-12', // Add right padding for the icon button
          error && 'ring-destructive/20 border-destructive',
          className
        )}
        placeholder={placeholder}
        disabled={disabled}
        {...props}
      />
      <button
        type="button"
        onClick={togglePasswordVisibility}
        disabled={disabled}
        className={cn(
          'absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-md transition-colors',
          'hover:bg-gray-100 dark:hover:bg-gray-800',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
          'disabled:opacity-50 disabled:cursor-not-allowed'
        )}
        aria-label={showPassword ? 'Hide password' : 'Show password'}
      >
        <img
          src={showPassword ? eyeCloseIcon : eyeOpenIcon}
          alt={showPassword ? 'Hide password' : 'Show password'}
          className="w-5 h-5"
        />
      </button>
    </div>
  )
}

export { PasswordInput }
