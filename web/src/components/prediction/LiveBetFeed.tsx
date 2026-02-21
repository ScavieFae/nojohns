/**
 * Live bet feed — terminal-styled scrolling feed of onchain prediction activity.
 * Shows BetPlaced, PoolCreated, and PoolResolved events in real-time.
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

function FeedLine({ event }: { event: FeedEvent }) {
  const time = formatTime(new Date(event.timestamp));
  const txShort = event.txHash.slice(0, 10);

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
          <span className="text-accent-green font-bold">{event.amount} MON</span>
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

export function LiveBetFeed() {
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
        <span className="w-2 h-2 bg-accent-green rounded-full animate-pulse" />
        <span className="text-xs font-mono font-bold text-gray-300">LIVE</span>
        <span className="text-xs text-gray-500">Prediction Market Activity</span>
        <span className="ml-auto text-xs text-gray-600 font-mono">
          {events.length} events
        </span>
      </div>

      {/* Feed */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="h-64 overflow-y-auto font-mono scrollbar-thin scrollbar-thumb-surface-600"
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
            <div className="text-center">
              <p className="text-gray-500 text-xs">Waiting for bets...</p>
              <p className="text-gray-600 text-xs mt-1">
                New activity will appear here in real-time
              </p>
            </div>
          </div>
        ) : (
          <div className="py-1">
            {events.map((event) => (
              <FeedLine key={event.id} event={event} />
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
