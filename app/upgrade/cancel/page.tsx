export default function UpgradeCancelPage() {
  return (
    <div className="min-h-screen text-zinc-100">
      <div className="container mx-auto px-4 pt-10 pb-10 max-w-xl">
        <h1 className="text-2xl font-bold mb-2">Checkout canceled</h1>
        <p className="text-zinc-400 mb-6">
          No charge was made. You can try again anytime.
        </p>
        <a href="/upgrade" className="text-sm text-indigo-400 hover:text-indigo-300 underline">Back to Upgrade</a>
      </div>
    </div>
  )
}
