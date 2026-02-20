import { useQuery } from '@tanstack/react-query';
import { fetchTracklets } from '../lib/api';

export function useTracklets(videoId: string | null) {
  return useQuery({
    queryKey: ['tracklets', videoId],
    queryFn: () => fetchTracklets(videoId!),
    enabled: !!videoId,
    staleTime: Infinity,
  });
}
