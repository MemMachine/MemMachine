import { DropdownMenuItem } from '@/components/ui/dropdown-menu';
import { useUserContext } from '@/contexts/UserContext';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  ChevronsUpDownIcon,
  LogOutIcon,
  UserIcon,
} from 'lucide-react';
import { FC } from 'react';
import { useTranslation } from 'react-i18next';

type MenuItem = 'settings' | 'logout' | 'profile';
type Side = 'top' | 'bottom' | 'left' | 'right';
type Align = 'start' | 'end' | 'center';

interface UserDropdownProps extends React.PropsWithChildren {
  menus?: MenuItem[];
  side?: Side;
  align?: Align;
}

export const UserDropdown: FC<UserDropdownProps> = (props) => {
  const { logout } = useUserContext();
  const { t } = useTranslation();

  const handleLogout = async () => {
    await logout();
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="cursor-pointer">
        <div className="flex items-center justify-between text-sm font-semibold cursor-pointer p-1">
          <div className="flex items-center gap-2">
            <div className="rounded-[50%] border w-6 h-6 flex items-center justify-center">
              <UserIcon className="w-6" />
            </div>
          </div>
          <ChevronsUpDownIcon className="w-4 ml-2" />
        </div>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side={props.side || 'bottom'}
        align={props.align || 'end'}
      >
        {/* <DropdownMenuItem
          className="cursor-pointer"
          onSelect={() => {
            navigate('/settings');
          }}
        >
          <SettingsIcon />
          {t('userDropdown.settings')}
        </DropdownMenuItem> */}
        {/* <DropdownMenuItem
          className="cursor-pointer"
          onSelect={() => {
            navigate('/profile');
          }}
        >
          <UserIcon />
          {t('userDropdown.profile')}
        </DropdownMenuItem> */}

        <DropdownMenuItem className="cursor-pointer" onSelect={handleLogout}>
          <LogOutIcon />
          {t('userDropdown.logout')}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
