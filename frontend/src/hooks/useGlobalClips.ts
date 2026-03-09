import { useQuery } from '@tanstack/react-query';
import { fetchGlobalClips } from '../lib/api';

export function useGlobalClips(videoId: string | null) {
  return useQuery({
    queryKey: ['global-clips', videoId],
    queryFn: () => fetchGlobalClips(videoId!),
    enabled: !!videoId,
    staleTime: Infinity,
  });
}
