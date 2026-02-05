interface MatchFilterProps {
  filterAddress: string;
  onFilterChange: (address: string) => void;
}

export function MatchFilter({ filterAddress, onFilterChange }: MatchFilterProps) {
  return (
    <div className="flex gap-3 items-center">
      <input
        type="text"
        placeholder="Filter by agent address..."
        value={filterAddress}
        onChange={(e) => onFilterChange(e.target.value)}
        className="bg-surface-700 border border-surface-600 rounded-lg px-4 py-2 text-sm font-mono text-gray-300 placeholder-gray-500 focus:outline-none focus:border-accent-blue w-full max-w-96"
      />
      {filterAddress && (
        <button
          onClick={() => onFilterChange("")}
          className="text-gray-500 hover:text-white text-sm transition-colors"
        >
          Clear
        </button>
      )}
    </div>
  );
}
