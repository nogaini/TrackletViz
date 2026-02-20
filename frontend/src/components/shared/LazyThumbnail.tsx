import { useEffect, useState } from 'react';
import { fetchTrackletThumbnail } from '../../lib/api';

/** Module-level cache: tracklet_id → base64 string (or '' if not found) */
const thumbnailCache = new Map<string, string>();
/** In-flight fetches to avoid duplicate requests */
const inFlight = new Map<string, Promise<string | null>>();

export function useThumbnail(trackletId: string | null): string | undefined {
  const [src, setSrc] = useState<string | undefined>(
    trackletId ? (thumbnailCache.get(trackletId) ?? undefined) : undefined,
  );

  useEffect(() => {
    if (!trackletId) return;
    const cached = thumbnailCache.get(trackletId);
    if (cached !== undefined) {
      setSrc(cached || undefined);
      return;
    }

    let current = true;
    let p = inFlight.get(trackletId);
    if (!p) {
      p = fetchTrackletThumbnail(trackletId);
      inFlight.set(trackletId, p);
    }
    p.then((val) => {
      inFlight.delete(trackletId);
      const result = val ?? '';
      thumbnailCache.set(trackletId, result);
      if (current) setSrc(result || undefined);
    });

    return () => { current = false; };
  }, [trackletId]);

  return src;
}

interface Props {
  trackletId: string;
  /** Pre-supplied base64 string; if provided, skips the network fetch */
  srcOverride?: string;
  className?: string;
  alt?: string;
}

/**
 * Renders a thumbnail image.
 * - If `srcOverride` is provided (e.g. from search results that already carry
 *   the thumbnail), it is used directly without a network request.
 * - Otherwise the thumbnail is fetched on-demand and cached for reuse.
 * - Shows a grey placeholder while loading or when unavailable.
 */
export default function LazyThumbnail({ trackletId, srcOverride, className = '', alt = '' }: Props) {
  const fetched = useThumbnail(srcOverride ? null : trackletId);
  const src = srcOverride || fetched;

  if (!src) {
    return (
      <div className={`bg-gray-700 flex items-center justify-center ${className}`}>
        <span className="text-[9px] text-gray-500">…</span>
      </div>
    );
  }

  return (
    <img
      src={`data:image/jpeg;base64,${src}`}
      alt={alt}
      className={className}
    />
  );
}
