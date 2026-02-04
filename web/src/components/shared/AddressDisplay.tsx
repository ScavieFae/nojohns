import { truncateAddress, explorerLink } from "../../lib/addresses";
import { getAgentName } from "../../lib/mockData";
import { USE_MOCK_DATA } from "../../config";

interface AddressDisplayProps {
  address: string;
  link?: boolean;
  full?: boolean;
}

export function AddressDisplay({ address, link = true, full = false }: AddressDisplayProps) {
  const display = full ? address : truncateAddress(address);
  const agentName = USE_MOCK_DATA ? getAgentName(address) : null;

  const content = (
    <span className="font-mono text-sm">
      {agentName ? (
        <>
          <span className="text-gray-200">{agentName}</span>
          <span className="text-gray-500 ml-1.5">{truncateAddress(address)}</span>
        </>
      ) : (
        display
      )}
    </span>
  );

  if (!link) return content;

  return (
    <a
      href={explorerLink("address", address)}
      target="_blank"
      rel="noopener noreferrer"
      className="hover:text-accent-blue transition-colors"
    >
      {content}
    </a>
  );
}
