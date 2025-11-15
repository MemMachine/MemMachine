import { Button } from '@/components/ui/button';
import { Link, NavLink } from 'react-router-dom';
import { UserDropdown } from '../user-dropdown';
import { useUserContext } from '@/contexts';
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from '@/components/ui/navigation-menu';
interface MenuItem {
  title: string;
  url: string;
  description?: string;
  icon?: React.ReactNode;
  items?: MenuItem[];
}

interface NavbarProps {
  logo?: {
    url: string;
    src: string;
    alt: string;
    title: string;
  };
  menu?: MenuItem[];
  auth?: {
    login: {
      title: string;
      url: string;
    };
    signup: {
      title: string;
      url: string;
    };
  };
}

const Navbar = (_props: NavbarProps) => {

  const {user} = useUserContext()

  const logo = {
    url: '/',
    src: 'https://deifkwefumgah.cloudfront.net/shadcnblocks/block/logos/shadcnblockscom-icon.svg',
    alt: 'logo',
    title: 'MemMachine',
  }

  const auth = {
    login: { title: 'Login', url: '/login' },
    signup: { title: 'Sign Up', url: '/signup' },
  }

  const menu = [
    { title: 'Home', url: '/' },
    {
      title: 'Help',
      url: '/help',
    }
  ]

  return (
    <div className="border-b">
      <section className="py-4 h-full flex items-center justify-between mx-auto px-4 sm:px-6">
      <div className="w-full">
        <nav className="justify-between flex">
          <div className="flex items-center gap-6">
            {/* Logo */}
            <Link to={logo.url} className="flex items-center gap-2">
              <img src={logo.src} alt={logo.alt} className="w-8 h-8 dark:invert" />
              <span className="text-lg font-semibold tracking-tighter">
                {logo.title}
              </span>
            </Link>
            <div className="flex items-center">
              <NavigationMenu viewport={false}>
                <NavigationMenuList>
                  {menu.map((item) => renderMenuItem(item))}
                </NavigationMenuList>
              </NavigationMenu>
            </div>
          </div>
          {!user && <div className="flex gap-2">
            <Button  size="sm">
              <Link to={auth.login.url}>{auth.login.title}</Link>
            </Button>
            <Button variant="outline" size="sm">
              <Link to={auth.signup.url}>{auth.signup.title}</Link>
            </Button>
          </div>}
          {user && <UserDropdown />}
        </nav>
      </div>
    </section>
    </div>
  );
};


const renderMenuItem = (item: MenuItem) => {
  // if (item.items) {
  //   return (
  //     <NavigationMenuItem key={item.title}>
  //       <NavigationMenuTrigger>{item.title}</NavigationMenuTrigger>
  //       <NavigationMenuContent>
  //         <div className="grid gap-2 md:w-[400px] lg:w-[500px] lg:grid-cols-[1fr_1fr]">

  //           {item.items.map((subItem) => (
  //             <NavigationMenuLink asChild key={subItem.title} className="w-80">
  //               <SubMenuLink item={subItem} />
  //             </NavigationMenuLink>
  //           ))}
        
  //         </div>
  //       </NavigationMenuContent> 
  //     </NavigationMenuItem>
  //   );
  // }

  return (
    <NavigationMenuItem key={item.title}>
      <NavigationMenuLink
        asChild
        className="px-4 [&[aria-current='page']]:bg-muted"
      >
        <NavLink 
          to={item.url}
        >
          {item.title}
        </NavLink>
      </NavigationMenuLink>
    </NavigationMenuItem>
  );
};

export default Navbar;
