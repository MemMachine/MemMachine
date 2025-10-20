import { useUserContext } from '@/contexts/UserContext';
import { Button } from './ui/button';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';

export function UserInfo() {
  const { user, loading, error } = useUserContext();
  const { t } = useTranslation();
  const navigate = useNavigate();

  if (loading) {
    return (
      <div className="flex items-center space-x-2">
        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
        <span className="text-sm text-muted-foreground">
          {t('common.loading')}
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-sm text-destructive">
        {t('common.error')}: {error.message}
      </div>
    );
  }

  if (!user) {
    return (
      <div className="flex items-center space-x-2">
        <span className="text-sm text-muted-foreground">
          {t('common.notLoggedIn')}
        </span>
        <Button variant="outline" size="sm" asChild>
          <a href="/login">{t('common.login')}</a>
        </Button>
      </div>
    );
  }

  return (
    <span
      onClick={() => {
        navigate('/wallet');
      }}
      className="text-sm font-medium text-primary mr-0 bg-primary/30 rounded-full px-2 py-1 cursor-pointer mr-3"
    >
      Name
    </span>
  );
}
