import {
  getActivities,
  getActivitiesByUsername,
} from '@/lib/actions/activity.actions';
import { getUsersByModel } from '@/lib/actions/user.actions';
import ListActivities from '@/components/ListActivities';
export const dynamic = 'force-dynamic';
async function Page({
  searchParams,
}: {
  searchParams: { [key: string]: string | undefined };
}) {
  const username = searchParams?.q ?? '';
  const activitiesFromUser = await getActivitiesByUsername(username);
  const users = await getUsersByModel();
  return (
    <div>
      <ListActivities
        key={'1'}
        activities={activitiesFromUser}
        users={users}
      />
    </div>
  );
}
export default Page;
