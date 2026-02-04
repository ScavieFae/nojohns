interface EmptyStateProps {
  title: string;
  description: string;
}

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="text-center py-16">
      <p className="text-gray-400 text-lg font-medium">{title}</p>
      <p className="text-gray-500 text-sm mt-2">{description}</p>
    </div>
  );
}
