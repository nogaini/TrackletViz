import { useQuery } from '@tanstack/react-query';
import { fetchTracklets } from '../lib/api';

const PAGE_SIZE = 10_000;

/**
 * Progressively loads all tracklets for a video in pages of PAGE_SIZE.
 *
 * Returns:
 *  - `data`: merged array of all loaded tracklets so far
 *  - `isLoading`: true while the first page is in flight
 *  - `isFetching`: true while any page is in flight
 *  - `total`: total_tracklets from video metadata (passed in so we know when done)
 */
export function useTracklets(videoId: string | null, totalTracklets?: number) {
  return useQuery({
    queryKey: ['tracklets', videoId],
    queryFn: async () => {
      if (!videoId) return [];

      const total = totalTracklets ?? Infinity;
      const pages: Awaited<ReturnType<typeof fetchTracklets>>[] = [];
      let offset = 0;

      while (offset < total) {
        const page = await fetchTracklets(videoId, {
          limit: PAGE_SIZE,
          offset,
          includeThumbnails: false,
        });
        pages.push(page);
        offset += page.length;

        // If we got fewer than a full page, we've reached the end
        if (page.length < PAGE_SIZE) break;
      }

      return pages.flat();
    },
    enabled: !!videoId,
    staleTime: Infinity,
  });
}
