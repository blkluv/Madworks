export default function TestPage() {
  return (
    <div className="min-h-screen bg-black text-white p-8">
      <h1 className="text-4xl font-bold mb-8">Test Layout</h1>
      
      <div className="bg-zinc-900/50 p-6 rounded-lg border border-zinc-800 mb-6">
        <h2 className="text-2xl font-semibold mb-4">Content Block</h2>
        <p className="text-zinc-300 mb-4">
          This is a test of the layout system. If this looks correct, the issue is likely in the main layout or component structure.
        </p>
        <div className="flex gap-4">
          <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-md transition-colors">
            Test Button
          </button>
          <button className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-md transition-colors">
            Secondary
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[1, 2, 3].map((item) => (
          <div key={item} className="bg-zinc-900/50 p-6 rounded-lg border border-zinc-800">
            <h3 className="text-xl font-medium mb-2">Card {item}</h3>
            <p className="text-zinc-400 text-sm">
              This is a test card to check grid layout and spacing.
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
