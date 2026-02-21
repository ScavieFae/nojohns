/**
 * Live bet feed — terminal-styled scrolling feed of onchain prediction activity.
 * Shows BetPlaced, PoolCreated, and PoolResolved events in real-time.
 *
 * Adapts layout: compact stacked lines in sidebar, full horizontal in wide containers.
 */

import { useEffect, useRef } from "react";
import { useLiveBetFeed, type FeedEvent } from "../../hooks/useLiveBetFeed";
import { explorerLink } from "../../lib/addresses";

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** Truncate MON amounts to 4 decimal places */
function shortMon(amount: string): string {
  const num = parseFloat(amount);
  if (isNaN(num)) return amount;
  if (num >= 1) return num.toFixed(2);
  if (num >= 0.01) return num.toFixed(4);
  return num.toFixed(4);
}

function FeedLine({ event, compact }: { event: FeedEvent; compact?: boolean }) {
  const time = formatTime(new Date(event.timestamp));
  const txShort = event.txHash.slice(0, 10);

  // Compact layout for sidebar
  if (compact) {
    if (event.type === "bet") {
      return (
        <a
          href={explorerLink("tx", event.txHash)}
          target="_blank"
          rel="noopener noreferrer"
          className="block px-2 py-1 hover:bg-surface-700/50 transition-colors text-[11px] leading-tight"
        >
          <div className="flex items-center justify-between">
            <span className="text-gray-600">{time}</span>
            <span className="text-gray-500 font-mono">{event.bettor}</span>
          </div>
          <div className="mt-0.5">
            <span className="text-accent-green font-bold">{shortMon(event.amount!)} MON</span>
            <span className="text-gray-500"> on </span>
            <span className={event.side === "A" ? "text-accent-green font-bold" : "text-purple-400 font-bold"}>
              P{event.side}
            </span>
          </div>
        </a>
      );
    }

    if (event.type === "pool_created") {
      return (
        <a
          href={explorerLink("tx", event.txHash)}
          target="_blank"
          rel="noopener noreferrer"
          className="block px-2 py-1 hover:bg-surface-700/50 transition-colors text-[11px] leading-tight"
        >
          <span className="text-yellow-500/80">Pool #{event.poolId} opened</span>
        </a>
      );
    }

    if (event.type === "pool_resolved") {
      return (
        <a
          href={explorerLink("tx", event.txHash)}
          target="_blank"
          rel="noopener noreferrer"
          className="block px-2 py-1 hover:bg-surface-700/50 transition-colors text-[11px] leading-tight"
        >
          <span className="text-blue-400/80">
            Pool #{event.poolId} resolved · {event.totalPool && shortMon(event.totalPool)} MON
          </span>
        </a>
      );
    }

    return null;
  }

  // Full-width layout
  if (event.type === "bet") {
    return (
      <a
        href={explorerLink("tx", event.txHash)}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-baseline gap-3 px-3 py-1 hover:bg-surface-700/50 transition-colors group"
      >
        <span className="text-gray-600 text-xs w-16 flex-shrink-0">{time}</span>
        <span className="text-gray-400 font-mono text-xs w-20 flex-shrink-0">
          {event.bettor}
        </span>
        <span className="flex-1 text-xs">
          <span className="text-white">bet </span>
          <span className="text-accent-green font-bold">{shortMon(event.amount!)} MON</span>
          <span className="text-white"> on </span>
          <span className={event.side === "A" ? "text-accent-green font-bold" : "text-purple-400 font-bold"}>
            Player {event.side}
          </span>
        </span>
        <span className="text-gray-600 text-xs flex-shrink-0">
          Pool #{event.poolId}
        </span>
        <span className="text-gray-700 text-xs flex-shrink-0 group-hover:text-gray-500">
          tx:{txShort}
        </span>
      </a>
    );
  }

  if (event.type === "pool_created") {
    return (
      <a
        href={explorerLink("tx", event.txHash)}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-baseline gap-3 px-3 py-1 hover:bg-surface-700/50 transition-colors group"
      >
        <span className="text-gray-600 text-xs w-16 flex-shrink-0">{time}</span>
        <span className="text-yellow-500 font-mono text-xs w-20 flex-shrink-0">
          system
        </span>
        <span className="flex-1 text-xs text-yellow-500/80">
          {event.text}
        </span>
        <span className="text-gray-600 text-xs flex-shrink-0">
          Pool #{event.poolId}
        </span>
        <span className="text-gray-700 text-xs flex-shrink-0 group-hover:text-gray-500">
          tx:{txShort}
        </span>
      </a>
    );
  }

  if (event.type === "pool_resolved") {
    return (
      <a
        href={explorerLink("tx", event.txHash)}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-baseline gap-3 px-3 py-1 hover:bg-surface-700/50 transition-colors group"
      >
        <span className="text-gray-600 text-xs w-16 flex-shrink-0">{time}</span>
        <span className="text-blue-400 font-mono text-xs w-20 flex-shrink-0">
          system
        </span>
        <span className="flex-1 text-xs text-blue-400/80">
          {event.text}
        </span>
        <span className="text-gray-600 text-xs flex-shrink-0">
          Pool #{event.poolId}
        </span>
        <span className="text-gray-700 text-xs flex-shrink-0 group-hover:text-gray-500">
          tx:{txShort}
        </span>
      </a>
    );
  }

  return null;
}

interface LiveBetFeedProps {
  compact?: boolean;
}

export function LiveBetFeed({ compact }: LiveBetFeedProps) {
  const { events, isLoading } = useLiveBetFeed();
  const scrollRef = useRef<HTMLDivElement>(null);
  const wasAtBottomRef = useRef(true);

  // Auto-scroll when new events arrive, but only if user was already at bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (wasAtBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [events]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    wasAtBottomRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
  };

  return (
    <div className="bg-surface-900 border border-surface-600 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-surface-700 bg-surface-800">
        <span className="w-2 h-2 bg-accent-green rounded-full animate-pulse flex-shrink-0" />
        <span className="text-xs font-mono font-bold text-gray-300">LIVE</span>
        {!compact && (
          <span className="text-xs text-gray-500">Prediction Market Activity</span>
        )}
        <span className="ml-auto text-xs text-gray-600 font-mono">
          {events.length}
        </span>
      </div>

      {/* Feed */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className={`overflow-y-auto font-mono scrollbar-thin scrollbar-thumb-surface-600 ${
          compact ? "h-48" : "h-64"
        }`}
      >
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="animate-spin w-5 h-5 border-2 border-accent-green border-t-transparent rounded-full mx-auto mb-2" />
              <p className="text-gray-500 text-xs">Scanning chain...</p>
            </div>
          </div>
        ) : events.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center px-2">
              <p className="text-gray-500 text-xs">Waiting for bets...</p>
              <p className="text-gray-600 text-xs mt-1">
                Activity appears here in real-time
              </p>
            </div>
          </div>
        ) : (
          <div className="py-1">
            {events.map((event) => (
              <FeedLine key={event.id} event={event} compact={compact} />
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-center px-3 py-1.5 border-t border-surface-700 bg-surface-800">
        <span className="text-[10px] text-gray-600 font-mono">
          NO JOHNS · Prediction Market · Monad
        </span>
      </div>
    </div>
  );
}
