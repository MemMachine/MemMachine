import { useSearchParams } from 'react-router-dom';

const useQueryParams = () => {
  const [searchParams] = useSearchParams();
  return Object.fromEntries([...searchParams]);
};

export default useQueryParams
