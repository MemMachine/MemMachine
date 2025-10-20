import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useUserContext } from '@/contexts/UserContext'
import { Loader2, User, Mail, Calendar, Key, Clock } from 'lucide-react'

export default function Home() {
  const { user, token, expiresAt, loading, error } = useUserContext()

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="flex items-center space-x-2">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span className="text-lg">Loading user information...</span>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <Card className="max-w-md mx-auto">
          <CardContent className="pt-6">
            <div className="text-center text-destructive">
              <p className="text-lg font-semibold mb-2">Error Loading User Information</p>
              <p className="text-sm">{error.message}</p>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="container mx-auto p-6">
        <Card className="max-w-md mx-auto">
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-lg font-semibold mb-2">No User Information Available</p>
              <p className="text-sm text-muted-foreground">Please log in to view your profile.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Format dates
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">User Profile</h1>
        <p className="text-muted-foreground mt-2">
          Complete information about your account
        </p>
      </div>

      <div className="grid gap-6">
        {/* Basic Information Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" />
              Basic Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-muted-foreground">User ID</label>
                <p className="text-lg font-semibold">#{user.id}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">Username</label>
                <p className="text-lg font-semibold">{user.username}</p>
              </div>
            </div>
            
            <div>
              <label className="text-sm font-medium text-muted-foreground flex items-center gap-1">
                <Mail className="h-4 w-4" />
                Email Address
              </label>
              <p className="text-lg">{user.email || '-'}</p>
            </div>
          </CardContent>
        </Card>
       
        {/* Authentication Token Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Key className="h-5 w-5" />
              Authentication Token
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium text-muted-foreground flex items-center gap-1">
                <Key className="h-4 w-4" />
                Access Token
              </label>
              <div className="mt-1 p-3 bg-muted rounded-md">
                <p className="text-sm font-mono break-all">
                  {token || '-'}
                </p>
              </div>
            </div>
            
            <div>
              <label className="text-sm font-medium text-muted-foreground flex items-center gap-1">
                <Clock className="h-4 w-4" />
                Token Expiration
              </label>
              {/* <div className="mt-1">
                {expiresAt ? (
                  <div className="flex items-center gap-2">
                    <p className="text-sm">{formatDate(expiresAt)}</p>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No expiration info available</p>
                )}
              </div> */}
            </div>

            {expiresAt && (
              <div className="mt-4 p-3 bg-muted rounded-md">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-medium">{formatDate(expiresAt)}</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

         {/* Account Timeline Card */}
         <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5" />
              Account Timeline
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-muted-foreground">Account Created</label>
                <p className="text-sm">{formatDate(user.created_at)}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">Last Updated</label>
                <p className="text-sm">{formatDate(user.updated_at)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
