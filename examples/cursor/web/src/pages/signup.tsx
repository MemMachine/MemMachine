import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { Link } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import { useUserContext } from '@/contexts/UserContext'
import { IRegisterReq } from '@/models/user'

const signupSchema = z.object({
  username: z.string().min(3, {
    message: 'Username must be at least 3 characters.',
  }).max(50, {
    message: 'Username must be at most 50 characters.',
  }),
  email: z.string().email({
    message: 'Please enter a valid email address.',
  }).optional().or(z.literal('')),
  password: z.string().min(6, {
    message: 'Password must be at least 6 characters.',
  }),
  confirmPassword: z.string().min(6, {
    message: 'Please confirm your password.',
  }),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
})

type SignupFormValues = z.infer<typeof signupSchema>

export default function Signup() {
  const { register, loading } = useUserContext()
  
  const form = useForm<SignupFormValues>({
    // @ts-ignore - zodResolver version compatibility issue
    resolver: zodResolver(signupSchema),
    defaultValues: {
      username: '',
      email: '',
      password: '',
      confirmPassword: '',
    },
  })

  const onSubmit = async (values: SignupFormValues) => {
    // Prepare registration data, removing confirmPassword and handling empty email
    const registrationData: IRegisterReq = {
      username: values.username,
      password: values.password,
      email: values.email || undefined,
    }
    
    await register(registrationData)
  }

  return (
    <div className="h-[calc(100vh-70px)] flex flex-col items-center justify-center">
      <Card className="overflow-hidden py-3 w-[440px] mx-auto">
        <CardContent className="p-0">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="p-6">
              <div className="flex flex-col gap-6">
                <div className="flex flex-col gap-3 items-center text-center">
                  <h1 className="text-2xl font-bold">Create your account</h1>
                  <p className="text-muted-foreground text-balance">
                    Join MemMachine and start managing your memories
                  </p>
                </div>
                
                <FormField
                  control={form.control}
                  name="username"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Username</FormLabel>
                      <FormControl>
                        <Input
                          placeholder="Enter your username"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <FormField
                  control={form.control}
                  name="email"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Email (Optional)</FormLabel>
                      <FormControl>
                        <Input
                          type="email"
                          placeholder="Enter your email"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <FormField
                  control={form.control}
                  name="password"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Password</FormLabel>
                      <FormControl>
                        <Input
                          type="password"
                          placeholder="Enter your password"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <FormField
                  control={form.control}
                  name="confirmPassword"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Confirm Password</FormLabel>
                      <FormControl>
                        <Input
                          type="password"
                          placeholder="Confirm your password"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <Button 
                  type="submit" 
                  className="w-full"
                  disabled={loading}
                >
                  {loading ? 'Creating account...' : 'Create account'}
                </Button>
                
                <div className="after:border-border relative text-center text-sm after:absolute after:inset-0 after:top-1/2 after:z-0 after:flex after:items-center after:border-t">
                  <span className="bg-card text-muted-foreground relative z-10 px-2">
                    Already have an account?
                  </span>
                </div>
                
                <div className="text-center text-sm">
                  <Link to="/login" className="underline underline-offset-4">
                    Sign in
                  </Link>
                </div>
              </div>
            </form>
          </Form>
        </CardContent>
      </Card>
    </div>
  )
}
