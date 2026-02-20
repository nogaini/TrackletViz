import type { VideoSummary, VideoMetadata, TrackletMetadata, SearchResult } from '../types/index';

export const fetchVideos = (): Promise<VideoSummary[]> =>
  fetch('/api/videos/').then(r => r.json()) as Promise<VideoSummary[]>;

export const fetchVideoMetadata = (id: string): Promise<VideoMetadata> =>
  fetch(`/api/videos/${id}`).then(r => r.json()) as Promise<VideoMetadata>;

/**
 * Fetch a page of tracklets for a video.
 *
 * - `limit=0` asks the backend for ALL tracklets (no pagination).
 * - `includeThumbnails=false` (default) strips thumbnail data from the payload.
 *   Thumbnails should be fetched individually via fetchTrackletThumbnail().
 */
export const fetchTracklets = (
  id: string,
  opts: { limit?: number; offset?: number; includeThumbnails?: boolean } = {},
): Promise<TrackletMetadata[]> => {
  const params = new URLSearchParams();
  if (opts.limit !== undefined) params.set('limit', String(opts.limit));
  if (opts.offset !== undefined) params.set('offset', String(opts.offset));
  if (opts.includeThumbnails) params.set('include_thumbnails', 'true');
  const qs = params.toString();
  return fetch(`/api/tracklets/${id}${qs ? `?${qs}` : ''}`).then(r => r.json()) as Promise<TrackletMetadata[]>;
};

/**
 * Fetch a single tracklet's thumbnail (base64 JPEG string).
 * Returns null if the tracklet doesn't exist.
 */
export const fetchTrackletThumbnail = async (trackletId: string): Promise<string | null> => {
  const r = await fetch(`/api/tracklets/${trackletId}/thumbnail`);
  if (!r.ok) return null;
  const data = (await r.json()) as { thumbnail_base64: string };
  return data.thumbnail_base64;
};

export const searchText = (
  video_id: string,
  query: string,
  limit = 20,
): Promise<SearchResult[]> =>
  fetch('/api/search/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_id, query, limit }),
  }).then(r => r.json()) as Promise<SearchResult[]>;

export const videoStreamUrl = (video_id: string): string =>
  `/api/videos/${video_id}/stream`;
