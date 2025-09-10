export default function UpgradeSuccessPage() {
  return (
    <div className="min-h-screen text-zinc-100">
      <div className="container mx-auto px-4 pt-10 pb-10 max-w-xl">
        <h1 className="text-2xl font-bold mb-2">Thank you! 🎉</h1>
        <p className="text-zinc-400 mb-6">
          Your Starter Kit purchase was successful. Your account will reflect the upgrade shortly.
        </p>
        <a href="/" className="text-sm text-indigo-400 hover:text-indigo-300 underline">Go back home</a>
      </div>
    </div>
  )
}
