import type { VideoSummary, VideoMetadata, TrackletMetadata, SearchResult } from '../types/index';

export const fetchVideos = (): Promise<VideoSummary[]> =>
  fetch('/api/videos/').then(r => r.json()) as Promise<VideoSummary[]>;

export const fetchVideoMetadata = (id: string): Promise<VideoMetadata> =>
  fetch(`/api/videos/${id}`).then(r => r.json()) as Promise<VideoMetadata>;

export const fetchTracklets = (id: string): Promise<TrackletMetadata[]> =>
  fetch(`/api/tracklets/${id}`).then(r => r.json()) as Promise<TrackletMetadata[]>;

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
