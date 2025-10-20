import { NotFound, IllustrationNotFound } from '@/components/ui/not-found'

function NotFoundPage() {
  return (
    <div className="relative flex flex-col w-full justify-center bg-background p-6 md:p-10">
      <div className="relative max-w-5xl mx-auto w-full">
        <IllustrationNotFound className="absolute inset-0 w-full h-[50vh] opacity-[0.04] dark:opacity-[0.03] text-foreground" />
        <NotFound
          title="Page not found"
          description="This page is not found."
        />
      </div>
    </div>
  )
}

export default NotFoundPage
