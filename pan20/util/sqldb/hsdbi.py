"""External package, locally added, providing repository and facade patterns."""
import importlib

import sqlalchemy as db
from sqlalchemy import func, orm
from sqlalchemy.dialects import sqlite


##########
# errors #
##########


class Error(Exception):
    pass


class NotFoundError(Exception):
    """A record was not found in the database.

    Attributes:
      pk: Dictionary, attr-value pairs representing the primary keys.
      table: the table searched.
    """

    def __init__(self, pk, table):
        self.pk = pk
        self.table = table


########
# base #
########


class RepositoryFacade:
    """Abstract base class for a Repository Facade.

    From the Wikipedia entry on "Facade pattern":
      'A facade is an object that provides a simplified interface to a
      larger body of code...'
      https://en.wikipedia.org/wiki/Facade_pattern
    In this case, we provide a single point of access for all Repository
    classes grouped in a conceptual unit, encapsulate the db connection,
    provide a commit() function for saving changes, and implement the magic
    methods __exit__ and __enter__ so this class is valid for use in a "with"
    statement.

    Collecting multiple repositories together might be viewed as an
    inefficiency: it is simple enough to initialize one Repository class as and
    when it is needed. Indeed, this is how I use repositories.

    The Facade comes in handy where we want to share database context between
    Repository classes.

    Implementation details for Repository Facade classes will differ with the
    database used. The intention is for a subclass to be defined for each
    such case. See MySQLRepositoryFacade and MongoRepositoryFacade, for example.
    """

    def __init__(self):
        """Create a new RepositoryFacade."""

    def __enter__(self):
        self.__init__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Implementation note - see:
        https://stackoverflow.com/questions/22417323/
        how-do-enter-and-exit-work-in-python-decorator-classes
        """
        self.dispose()

    def dispose(self):
        """Close the connection for disposal of the RepositoryFacade."""
        raise NotImplementedError()


class Repository:
    """Abstract Repository class

    The docstrings here will indicate the intention of the functions and their
    arguments. Child classes can extend these in part, requiring new arguments.

    The base class remains agnostic as to return types as this will vary across
    databases.

    The exception is the exists() function, which is implemented here.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __enter__(self):
        self.__init__(**self._kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Implementation note - see:
        https://stackoverflow.com/questions/22417323/
        how-do-enter-and-exit-work-in-python-decorator-classes
        """
        self.dispose()

    def add(self, item):
        """Add one or more items to the database.

        Args:
          items: an object or list of objects to add.
        """
        raise NotImplementedError()

    def all(self, projection=None):
        """Get all records in the database.

        Args:
          projection: List, optional, of attributes to project.
        """
        raise NotImplementedError()

    def commit(self):
        """Save changes to the database."""
        raise NotImplementedError()

    def count(self):
        """Get a count of how many records are in this table/collection."""
        raise NotImplementedError()

    def delete(self, items=None, **kwargs):
        """Delete item(s) from the database.

        May either specify an object or list of objects to delete, or
        keyword arguments for primary key identification.

        Args:
          items: an object or list of objects to delete.
          kwargs: can specify primary key attribute name(s) and value(s).
        """
        raise NotImplementedError()

    def dispose(self):
        """Dispose of this Repository."""
        raise NotImplementedError()

    def exists(self, **kwargs):
        """Check if a record exists.

        Pass the primary key values in as keyword arguments.

        Returns:
          Boolean indicating if the record exists.
        """
        # NOTE: to project here or not?
        return len(list(self.search(**kwargs))) > 0

    def get(self, expect=True, projection=None, **kwargs):
        """Get an item from the database.

        Primary key values need to be passed as keyword arguments.

        Arguments:
          expect: Bool, whether to throw an error if not found.
          projection: List, optional, of attributes to project.
          kwargs: can specify the primary key attribute name(s) and value(s).
        """
        raise NotImplementedError()

    def search(self, projection=None, **kwargs):
        """Search for records in the database.

        Specify the search attributes and values as keyword arguments.

        NOTE: This must return a generator or list of records - this is
        necessary for the default implementation of the exists() method. Any
        change to this means you should re-implement exists().

        Args:
          projection: List, optional, of attributes to project.
          kwargs: attribute name(s) and value(s) to search on.

        Returns:
          Generator or list of records.
        """
        raise NotImplementedError()


#######
# sql #
#######


# Static Functions


def create_sql_session(connection_string):
    """Create a SQLAlchemy session from a connection string.

    Args:
      connection_string: String.

    Returns:
      sqlalchemy.orm.session.Session object.
    """
    engine = db.create_engine(connection_string)
    session = orm.sessionmaker(bind=engine)()
    return session


def print_sql(query, dialect=sqlite):
    print(query.statement.compile(dialect=dialect,
                                  compile_kwargs={'literal_binds': True}))


# SQL Implementations


class SQLFacade(RepositoryFacade):
    """Facade for SQL repositories.

    Attributes:
      session: the sqlalchemy session wrapped by this facade. There is a
        question of whether to expose this. It was decided to expose it because
        it will enable flexibility since consumers can directly use it if
        convenient, providing better extensibility.
    """

    def __init__(self, connection_string):
        """Create a new RepositoryFacade.

        This will create the self._engine and self.session variables from
        the passed connection string. It also saves self._connection_string
        for reference.

        Args:
          connection_string: String, the connection string to the database.
        """
        super(SQLFacade, self).__init__()
        self._connection_string = connection_string
        self.session = create_sql_session(connection_string)

    def __enter__(self):
        SQLFacade.__init__(self, self._connection_string)
        return self

    def commit(self):
        """Save changes to the database."""
        self.session.commit()

    def delete_all_records(self):
        for attr, value in self.__dict__.items():
            if isinstance(value, SQLRepository):
                value.delete_all_records()

    def dispose(self):
        """Dispose of this class - close the database connection."""
        self.session.close()


class SQLRepository(Repository):
    """Generic wrapper for db access methods for a table."""

    def __init__(self, primary_keys, class_type, orm_module,
                 connection_string=None, session=None, **kwargs):
        """Create a new Repository.

        Args:
          class_type: Type, the type of the object this repository will handle.
            It should be one of the orm classes.
          orm_module: String, the name of the module containing orm classes
            that this Repository will work on. E.g. 'db.orm'.
          primary_keys: List of strings, identify the attribute names that
            represent the primary keys for this class.
          connection_string: String, optional, but must pass one of either
            connection_string or session.
          session: SQLAlchemy session object, optional, but must pass one of
            either connection_string or session.
        """
        super(SQLRepository, self).__init__(**kwargs)
        self._class_type = class_type
        self._orm_module = orm_module
        self._orm_module_key = orm_module.replace('.', '_')
        globals()[self._orm_module_key] = importlib.import_module(orm_module)
        self._primary_keys = primary_keys
        self._connection_string = connection_string
        self._session = session
        if connection_string:
            self._connection_string = connection_string
            self._session = create_sql_session(self._connection_string)
        elif session:
            self._session = session
        else:
            raise ValueError('You must pass either a session or '
                             'connection string.')

    def __enter__(self):
        self.__init__(self._primary_keys, self._class_type, self._orm_module,
                      self._connection_string, self._session, **self._kwargs)
        return self

    def add(self, items):
        """Add one or more items to the database.

        Args:
          items: one or more objects of the intended type; can be a list or
            a single object.
        """
        if isinstance(items, list):
            for item in items:
                self._session.add(item)
        else:
            self._session.add(items)

    def add_or_get(self, keys, other_attrs=None):
        if not self.exists(**keys):
            if other_attrs:
                attrs = {**keys, **other_attrs}
            else:
                attrs = keys
            item = self._class_type(**attrs)
            self.add(item)
            return item
        else:
            return self.get(**keys)

    def all(self, projection=None):
        """Retrieve all items of this kind from the database.

        Args:
          projection: List, optional, of attributes to project.

        Returns:
          List of items of the relevant type; or Tuple of values if projected.
        """
        query = self._session.query(self._class_type)
        if projection:
            query = self.project(query, projection)
        return query.all()

    def commit(self):
        """Commit changes to the database."""
        self._session.commit()

    def count(self):
        """Count the number of records in the table.

        Returns:
          Integer, the number of records in the table.
        """
        return self._session\
            .query(func.count(eval('%s.%s.%s' % (self._orm_module_key,
                                                 self._class_type.__name__,
                                                 self._primary_keys[0]))))\
            .scalar()

    def delete(self, items=None, **kwargs):
        """Delete item(s) from the database.

        Can either specify items directly as a single object of the expected
        type, or as a list of such objects; or specify a record to delete
        by primary key via keyword arguments.

        Args:
          items: one or more objects of the intended type; can be a list or
            a single object.
          kwargs: can specify the primary key name(s) and value(s).

        Raises:
          ValueError: if neither items nor keyword arguments are specified.
          ValueError: if items are not specified and any primary key is
            missing from the keyword arguments.
        """
        if not items and len(kwargs) == 0:
            raise ValueError('You must specify either items or kwargs.')
        if items:
            if isinstance(items, list):
                for item in items:
                    self._session.delete(item)
            else:
                self._session.delete(items)
        else:
            for pk in self._primary_keys:
                if pk not in kwargs.keys():
                    raise TypeError('Missing keyword argument: %s' % pk)
            self._session.delete(self.get(**kwargs))

    def delete_all_records(self):
        """Delete all records from this table.

        Returns:
          Integer, the number of records deleted.
        """
        return self._session.query(self._class_type).delete()

    def dispose(self):
        """Dispose of the database connection."""
        self._session.close()

    def get(self, expect=True, projection=None, debug=False, **kwargs):
        """Get an item from the database.

        Pass the primary key values in as keyword arguments.

        Args:
          expect: whether or not to expect the result. Will raise an exception
            if not found if True.
          projection: List of String attribute names to project, optional.
          kwargs: can specify the primary key attribute name(s) and value(s).
          debug: Boolean, if true will print the SQL of the query for
            debugging purposes.

        Raises:
          TypeError: if any primary key values are missing from the keyword
            arguments received.
          NotFoundError: if the item is not found in the database and the expect
            flag is set to True.
        """
        for pk in self._primary_keys:
            if pk not in kwargs.keys():
                raise TypeError('Missing keyword argument: %s' % pk)
        query = self._session.query(self._class_type)
        for attr, value in kwargs.items():
            query = query.filter(
                eval('%s.%s.%s == "%s"'
                     % (self._orm_module_key,
                        self._class_type.__name__, attr, value)))
        if projection:
            query = self.project(query, projection)
        if debug:
            print_sql(query)
        result = query.one_or_none()
        if not result and expect:
            raise NotFoundError(pk=kwargs, table=self._class_type)
        if projection:
            return result[0]
        else:
            return result

    def project(self, query, projection, debug=False):
        """Perfoms a projection on the given query.

        Args:
          query: the query object to project.
          projection: List of String attributes to project.
          debug: Boolean, if true will print the SQL of the query for
            debugging purposes.

        Returns:
          Query object.

        Raises:
          ValueError if projection is not a list. It is easy to pass a string
            here, so we will catch this case for quick debugging.
        """
        if not isinstance(projection, list):
            raise ValueError('projection must be a list.')
        projection = ['%s.%s.%s' % (self._orm_module_key,
                                    self._class_type.__name__, attr)
                      for attr in projection]
        query = query.with_entities(eval(', '.join(projection)))
        if debug:
            print_sql(query)
        return query

    def search(self, projection=None, debug=False, **kwargs):
        """Attempt to get item(s) from the database.

        Pass whatever attributes you want as keyword arguments.

        Args:
          projection: List of String attribute names to project, optional.
          debug: Boolean, if true will print the SQL of the query for
            debugging purposes.

        Returns:
          List of matching results; or Tuple if projected.
        """
        query = self._session.query(self._class_type)
        for attr, value in kwargs.items():
            query = query.filter(
                eval('%s.%s.%s == "%s"'
                     % (self._orm_module_key,
                        self._class_type.__name__,
                        attr,
                        value)))
        if projection:
            query = self.project(query, projection)
        if debug:
            print_sql(query)
        return query.all()
